import numpy as np
import laspy
from tqdm import tqdm
from deepforest import main
from samgeo import SamGeo


class ForestLidar:
    def __init__(self):
        '''Class constructor

        inizialize DeepForest and SAM model

        Attributes of a ForestLidar object:
        
        deep_forest : deepforest.main.deepforest
            DeepForest model

        sam : samgeo.samgeo.SamGeo
            SAM model loaded by using samgeo package
        '''

        self.deep_forest = main.deepforest()
        self.sam = SamGeo(model_type='vit_h', automatic=False, sam_kwargs=None) 

        self.deep_forest.load_model(model_name = 'weecology/deepforest-tree', revision='main')


    def _cloud_to_image(self, points, resolution):
        ''' Function projecting in 2D the LiDAR data

        Parameters:
        ------------
        points : numpy.ndarray
            numpy array of the LiDAR points, it must have shape [N_points, 6], 6 stands for [x,y,z,r,g,b]

        resolution : float
            resolution parameter controlling the output image dimension


        Returns:
        -----------
        image : numpy.ndarray
            2D RGB image resulting from the projection of the LiDAR data
        '''

        if points.shape[1] != 6:
            raise ValueError('2D dimension of input array must have 6 values: [x,y,z,r,g,b]')
        
        minx, maxx = np.min(points[:, 0]), np.max(points[:, 0])
        miny, maxy = np.min(points[:, 1]), np.max(points[:, 1])

        width = int((maxx - minx) / resolution) + 1
        height = int((maxy - miny) / resolution) + 1

        image = np.zeros((height, width, 3), dtype=np.uint8)

        for point in tqdm(points):
            x, y, z, r, g, b = point

            pixel_x = int((x - minx) / resolution)
            pixel_y = int((maxy - y) / resolution)
            
            image[pixel_y, pixel_x] = np.array([r, g, b])


        return image
    


    def _image_to_cloud(self, points, image, resolution):
        ''' Function restoring the 3D dimension taking as input the segmented 2D image and the LiDAR points

        Parameters:
        ------------
        points : numpy.ndarray
            numpy array of the LiDAR points, it must have shape [N_points, 6], 6 stands for [x,y,z,r,g,b]

        image : numpy.ndarray
            image containint segmenting masks resulting from the action of the model

        resolution : float
            resolution parameter controlling the projected image dimension, must be the same as _cloud_to_image


        Returns:
        -----------
        segment_ids : numpy.array
            1D array containing the labels (1/0 for 'tree'/'non tree') for each point
        '''

        minx, maxx = np.min(points[:, 0]), np.max(points[:, 0])
        miny, maxy = np.min(points[:, 1]), np.max(points[:, 1])

        segment_ids = []
        image = np.asarray(image)

        for point in tqdm(points):
            x, y, z, *_ = point

            pixel_x = int((x - minx) / resolution)
            pixel_y = int((maxy - y) / resolution)

            if not (0 <= pixel_x < image.shape[1]) or not (0 <= pixel_y < image.shape[0]):
                segment_ids.append(-1)
                continue

            rgb = tuple(image[pixel_y, pixel_x])

            if rgb == (0, 240, 0):
                id = 1
            else: 
                 id = 0

            segment_ids.append(id)

        return segment_ids
    



    def classify_lidar(self, las_path, resolution = 0.4, window_size = 60, patch_overlap = 0.25):
        '''Classification function. It classifies the tree points in a LiDAR point data cloud

        Parameters:
        ------------
        las_path : str
            path to the input .las file containing LiDAR data

        resolution : float
            resolution parameter controlling the projected image dimension

        window_size : float
            size of the window, in meters, with which the 2D image is batched

        patch_overlap : float
            percentage of windows patch overlap


        Returns:
        -----------
        tree_cloud : LasData
            a new file .las containing the dimension 'tree_labels' in which are stored the point labels
        '''

        las = laspy.read(las_path)

        cloud = las.points

        np_cloud = np.vstack((cloud.x, cloud.y, cloud.z, cloud.red / 255.0, cloud.green / 255.0, cloud.blue / 255.0)).transpose()

        print('\n Projecting into 2D image')
        image = self._cloud_to_image(np_cloud, resolution)


        print('\n SAM setting image \n')
        self.sam.set_image(image)


        # Computing the correct patch_size, given in pixel, corresponding to the given window_size (in meters)
        minx, maxx = np.min(np_cloud[:, 0]), np.max(np_cloud[:, 0])
        width = int((maxx - minx) / resolution) + 1

        meters_per_pixel = (maxx-minx)/width

        patch_size = window_size/meters_per_pixel
        


        predicted_raster = self.deep_forest.predict_tile(image = image, patch_size = int(patch_size), patch_overlap = patch_overlap)


        box = []
        for i in range(len(predicted_raster['xmin'])):
            box.append([predicted_raster['xmin'][i], predicted_raster['ymin'][i], predicted_raster['xmax'][i], predicted_raster['ymax'][i]])


        mask_color = np.zeros_like(image)

        print('\n SAM acting')

        for i in tqdm(range(len(box))):
            results = self.sam.predict(boxes = [box[i]], return_results = True)
            mask_color[results[0][2]] = [0, 240, 0] 

        print('\n Reconverting in 3d point cloud')
        labels_tree = self._image_to_cloud(np_cloud, mask_color, resolution = resolution)


        tree_cloud = laspy.LasData(las.header)
        tree_cloud.points = las.points

        tree_cloud.add_extra_dim(laspy.ExtraBytesParams(name="tree_labels", type=np.int32))
        tree_cloud.tree_labels = labels_tree

        return tree_cloud