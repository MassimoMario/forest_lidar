import argparse

from scripts.utils import set_type
from scripts.forest_lidar_class import ForestLidar


def main():
    ''' Main function to parse command-line arguments, then simulate a flock dynamic

    Command-line settings:

    Parameters
    ----------
    path : str
        path to the .las file containing LiDAR data

    resolution : float
        resolution parameter to project LiDAR points in a 2D image, default is 0.4

    window_size : float
        size of the window, in meters, to batch the image, default is 60

    patch_overlap : float
        percentage of windows patch overlap, to segment the image, default is 0.25

    save : bool
        save option

    save_path : str
        path to write the resulting .las file


    Returns
    -------
    None
    '''
    
    parser = argparse.ArgumentParser(
        description = "Classify tree points in a given .las file")
    
    parser.add_argument(
        "--path",
        type = str,
        help = "Path to .las file containing LiDAR data"
    )

    parser.add_argument(
        "--resolution",
        type = float,
        default = 0.4,
        help = "Resolution parameter to project LiDAR points in a 2D image"
    )


    parser.add_argument(
        "--window_size",
        type = float,
        default = 60,
        help = "Size of the window, in meters, to batch the image"
    )

    parser.add_argument(
        "--patch_overlap",
        type = float,
        default = 0.25,
        help = "Percentage of windows patch overlap, to segment the image"
    )


    parser.add_argument(
        "--save",
        type = bool,
        default = True,
        help = "Save option"
    )

    parser.add_argument(
        "--save_path",
        type = str,
        default = 'tree_cloud.las',
        help = "Path of the saved .las file"
    )




    args = parser.parse_args()

    forest_lidar = ForestLidar()

    tree_cloud = forest_lidar.classify_lidar(las_path = args.path, 
                                resolution = args.resolution, 
                                window_size = args.window_size, 
                                patch_overlap = args.patch_overlap)
    
    if args.save:
        tree_cloud.write(args.save_path)
    



    
if __name__ == '__main__':
    main()