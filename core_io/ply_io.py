import numpy as np


def save_pointcloud_to_ply(points_pos, points_color, ply_file_path):
    """
    Save the point cloud to ply file
    :param points_pos: 2D array of point clouds, dimension (num_points, 3)
    :param points_color: 2D array of point cloud color, (R, G, B) with value range [0, 1] for each component
    :param ply_file_path: the output file path
    """
    num_points = points_pos.shape[0]
    with open(ply_file_path, 'w', encoding='ascii') as ply_file:

        # Write headers
        headers = ["ply\n",
                  "format ascii 1.0\n",
                  "element face 0\n",
                  "property list uchar int vertex_indices\n",
                  "element vertex %d\n" % num_points,
                  "property float x\n",
                  "property float y\n",
                  "property float z\n",
                  "property uchar diffuse_red\n",
                  "property uchar diffuse_green\n",
                  "property uchar diffuse_blue\n",
                  "end_header\n"]

        for header in headers:
            ply_file.write(header)

        # Write point position and color
        for pt_idx in range(0, num_points):
            pt_pos = points_pos[pt_idx]
            pt_color = 255 * points_color[pt_idx]
            ply_file.write("%f %f %f %d %d %d\n" % (pt_pos[0], pt_pos[1], pt_pos[2], int(pt_color[0]), int(pt_color[1]), int(pt_color[2])))


def load_pointcloud_from_ply(ply_file_path):
    """
    Load the point cloud from ply file
    :param ply_file_path: ply file path
    :return: point cloud positions and colors
    """
    points_pos, points_color = None, None

    with open(ply_file_path, 'r', encoding='ascii') as seq_file:

        start_flag = False
        current_idx = 0

        # Read line-by-line to parsing attribute files
        while 1:
            lines = seq_file.readlines(100000)
            if not lines:
                break
            for line in lines:
                if line.startswith("element vertex"):
                    num_points = int(line.split(" ")[-1].strip())
                    points_pos = np.zeros((num_points, 3), dtype=np.float32)
                    points_color = np.zeros((num_points, 3), dtype=np.float32)
                    continue
                elif line.startswith("end_header"):
                    start_flag = True
                    continue

                if start_flag:
                    tokens = line.split(" ")
                    tokens[-1] = tokens[-1].strip()
                    x, y, z = float(tokens[0]), float(tokens[1]), float(tokens[2])
                    r, g, b = float(tokens[3]), float(tokens[4]), float(tokens[5])
                    points_pos[current_idx, :] = np.asarray((x, y, z), dtype=np.float32)
                    points_color[current_idx, :] = np.asarray((r, g, b), dtype=np.float32) / 255.0
                    current_idx += 1

    return points_pos, points_color
