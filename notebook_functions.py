"""
Functions in this file have all been defined in the notebooks. 
This file serves to allow subsequent notebooks to import 
functionality and reduce code duplication.
"""

import cv2
import numpy as np 
import ipyvolume as ipv

def calibrate_cameras() :
    """ Calibrates cameras from chessboard images. 
    
    Returns: 
        images (list[np.ndarray]): Images containing the chessboard.
        intrinsics (np.ndarray): An upper triangular 4x4 full-rank matrix containing camera intrinsics.
        distortions (np.ndarray): Radial distortion coefficients.
        rotation_vectors (list[np.ndarray]): Rodrigues rotation vectors.
        translation_vectors (list[np.ndarray]): Translation vectors.
        object_points: (np.ndarray): A (4, 54) point array, representing the [x,y,z,w]
            of 54 chessboard points (homogenous coordiantes).
    """
    images = list() 
    
    # Read images
    for i in range(11): 
        img = cv2.imread(f'./images/{i}.jpg')
        img = cv2.resize(img, None, fx=0.25, fy=0.25)
        images.append(img) 

    # The default opencv chessboard has 6 rows, 9 columns 
    shape = (6, 9) 

    # List to store vectors of 3D world points for every checkerboard image
    object_points_all = []
    
    # List to store vectors of 2D projected points for every checkerboard image
    image_points_all = [] 
    
    # Flags for chessboard corner search. Taken from opencv docs.
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    
    # Criteria for termination of the iterative corner refinement. Taken from opencv docs.
    refinement_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # List for images in which chessboard is be found by search
    images_filtered = list()
    
    # Object points in a single image. Simply a row iterated list of z=0 3d points. 
    # E.g. [[0. 0. 0.] [1. 0. 0.] ... [0, 1, 0], [1, 1, 0], ... ]
    object_points = np.zeros((1, shape[0] * shape[1], 3), np.float32)
    object_points[0, :, :2] = np.mgrid[0:shape[0], 0:shape[1]].T.reshape(-1, 2)
    
    # For each image, store the object points and image points of chessboard corners.
    for idx, image in enumerate(images): 
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        succes, corners = cv2.findChessboardCorners(image=gray, 
                                                    patternSize=shape,
                                                    flags=flags) 
        if succes:
            images_filtered.append(image)
            corners = cv2.cornerSubPix(image=gray, 
                                       corners=corners, 
                                       winSize=(11,11), 
                                       zeroZone=(-1,-1), 
                                       criteria=refinement_criteria)
            object_points_all.append(object_points)
            image_points_all.append(corners)    
            
    images = images_filtered
    
    # Calibrate the cameras by using the 3D <-> 2D point correspondences.
    ret, intrinsics, distortions, rotation_vectors, translation_vectors = cv2.calibrateCamera(object_points_all, image_points_all, gray.shape[::-1], None, None)
    
    # Make intrinsic matrix 4x4 full-rank to ease manipulation.
    intrinsics = np.hstack([intrinsics, np.zeros((3, 1))])
    intrinsics = np.vstack([intrinsics, [[0, 0, 0, 1]]])

    # Convert chessboard object points to homogeneous coordinates to ease later use.
    object_points = object_points[0].reshape((-1, 3)).T
    object_points = np.vstack([object_points, np.ones((1, object_points.shape[1]))])
    
    return images, intrinsics, distortions, rotation_vectors, translation_vectors, object_points
    
images, intrinsics, distortions, rotation_vectors, translation_vectors, object_points = calibrate_cameras()

def extrinsics_from_calibration(rotation_vectors, translation_vectors):
    """ Calculates extrinsic matrices from calibration output. 
        
    Args: 
        rotation_vectors (list[np.ndarray]): Rodrigues rotation vectors.
        translation_vectors (list[np.ndarray]): Translation vectors.
    Returns: 
        extrinsics (list[np.ndarray]): A list of camera extrinsic matrices.
            These matrices are 4x4 full-rank.
    """
    
    rotation_matrices = list() 
    for rot in rotation_vectors:
        rotation_matrices.append(cv2.Rodrigues(rot)[0]) 

    extrinsics = list()
    for rot, trans in zip(rotation_matrices, translation_vectors): 
        extrinsic = np.concatenate([rot, trans], axis=1)
        extrinsic = np.vstack([extrinsic, [[0,0,0,1]]]) 
        extrinsics.append(extrinsic)
    
    return extrinsics

def camera_centers_from_extrinsics(extrinsics):
    """ Calculates camera centers from extrinsic matrices. 
    
    Args: 
        extrinsics (list[np.ndarray]):  A list of camera extrinsic matrices.
    Returns: 
        camera_centers (list[np.ndarray]): Homogenous coordinates of camera centers in 
            3D world coordinate frame.
    """
    camera_centers = list() 

    for extrinsic in extrinsics:
        rot = extrinsic[:3, :3]
        trans = extrinsic[:3, 3]
        center = -rot.T @ trans
        center = np.append(center, 1)
        camera_centers.append(center)
    
    return camera_centers

extrinsics = extrinsics_from_calibration(rotation_vectors, translation_vectors)
camera_centers = camera_centers_from_extrinsics(extrinsics)
cam_sphere_size = 1

def init_3d_plot():
    """ Initializes a ipyvolume 3d plot and centers the 
        world view around the center of the chessboard. """
    chessboard_x_center = 2.5
    chessboard_y_center = 4
    fig = ipv.pylab.figure(figsize=(15, 15), width=800)
    ipv.xlim(2.5 - 30, 2.5 + 30)
    ipv.ylim(4 - 30, 4 + 30)
    ipv.zlim(-50, 10)
    ipv.pylab.view(azimuth=40, elevation=-150)

    return fig

def plot_chessboard(object_points):
    """ Plots a 3D chessboard and highlights the 
        objects points with green spheres. """
    
    img = cv2.imread('./images/chessboard.jpg')
    img_height, img_width, _ = img.shape
    chessboard_rows, chessboard_cols = 7, 10
    xx, yy = np.meshgrid(np.linspace(0, chessboard_rows, img_height), 
                         np.linspace(0, chessboard_cols, img_width)) 
    zz = np.zeros_like(yy)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255
    # -1 is used as the start of the board images x and y coord, 
    # such that the first inner corner appear as coord (0, 0, 0) 
    ipv.plot_surface(xx-1, yy-1, zz, color=cv2.transpose(img))
    xs, ys, zs, _ = object_points
    ipv.scatter(xs, ys, zs, size=1, marker='sphere', color='lime')
    
# Visual dimension of the camera in the 3D plot. 
height, width, _ = images[0].shape
camera_aspect_ratio = width / height
# A length of 1 corresponds to the length of 1 chessboard cell.
# This is because a chessboard points have been defined as such.
# Set height of camera viewport to 1.
vis_cam_height = 1 
vis_cam_width = vis_cam_height * camera_aspect_ratio
wire_frame_depth = 1.2

def plot_camera_wireframe(cam_center, vis_scale, inv_extrinsic, color='blue', cam_sphere_size=1):
    """ Plots the 'viewport' or 'wireframe' for a camera. """
    x, y, z = cam_center[:3]
    
    # Get left/right top/bottom wireframe coordinates
    # Use the inverse of the camera's extrinsic matrix to convert 
    # coordinates relative to the camera to world coordinates.
    lt = inv_extrinsic @ np.array((-vis_cam_width/2, -vis_cam_height/2, wire_frame_depth, 1/vis_scale)) * vis_scale
    rt = inv_extrinsic @ np.array((vis_cam_width/2,  -vis_cam_height/2, wire_frame_depth, 1/vis_scale)) * vis_scale
    lb = inv_extrinsic @ np.array((-vis_cam_width/2,  vis_cam_height/2, wire_frame_depth, 1/vis_scale)) * vis_scale
    rb = inv_extrinsic @ np.array((vis_cam_width/2,   vis_cam_height/2, wire_frame_depth, 1/vis_scale)) * vis_scale

    # Connect camera projective center to wireframe extremities
    p1 = ipv.plot([x, lt[0]], [y, lt[1]], [z, lt[2]], color=color)
    p2 = ipv.plot([x, rt[0]], [y, rt[1]], [z, rt[2]], color=color)
    p3 = ipv.plot([x, lb[0]], [y, lb[1]], [z, lb[2]], color=color)
    p4 = ipv.plot([x, rb[0]], [y, rb[1]], [z, rb[2]], color=color)
    
    # Connect wireframe corners with a rectangle
    p5 = ipv.plot([lt[0], rt[0]], [lt[1], rt[1]], [lt[2], rt[2]], color=color)
    p6 = ipv.plot([rt[0], rb[0]], [rt[1], rb[1]], [rt[2], rb[2]], color=color)
    p7 = ipv.plot([rb[0], lb[0]], [rb[1], lb[1]], [rb[2], lb[2]], color=color)
    p8 = ipv.plot([lb[0], lt[0]], [lb[1], lt[1]], [lb[2], lt[2]], color=color)
    
    p9 = ipv.scatter(np.array([x]), np.array([y]), np.array([z]), size=cam_sphere_size, marker="sphere", color=color)
    return [p1, p2, p3, p4, p5, p6, p7, p8, p9] 

def plot_picture(image, inv_extrinsic, vis_scale):
    """ Plots a real world image its respective 3D camera wireframe. """ 
    image = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    image = cv2.resize(image, None, fx=0.1, fy=0.1) / 255
    img_height, img_width, _ = image.shape

    xx, yy = np.meshgrid(np.linspace(-vis_cam_width/2 * vis_scale,  vis_cam_width/2 * vis_scale,  img_width), 
                         np.linspace(-vis_cam_height/2 * vis_scale, vis_cam_height/2 * vis_scale, img_height))
    zz = np.ones_like(yy) * wire_frame_depth * vis_scale
    coords = np.stack([xx, yy, zz, np.ones_like(zz)]) 
    coords = coords.reshape(4, -1) 
     
    # Convert canera relative coordinates to world relative coordinates
    coords = inv_extrinsic @ coords
    xx, yy, zz, ones = coords.reshape(4, img_height, img_width) 
    return ipv.plot_surface(xx, yy, zz, color=image)
    
def update_camera_wireframe(cam_center, vis_scale, inv_extrinsic, old_plot):
    """ Updates the camera wireframe. This allows for animating the wireframe. """
    [p1, p2, p3, p4, p5, p6, p7, p8, p9] = old_plot
    x, y, z = cam_center[:3]
    
    lt = inv_extrinsic @ np.array((-vis_cam_width/2, -vis_cam_height/2, wire_frame_depth, 1/vis_scale)) * vis_scale
    rt = inv_extrinsic @ np.array((vis_cam_width/2,  -vis_cam_height/2, wire_frame_depth, 1/vis_scale)) * vis_scale
    lb = inv_extrinsic @ np.array((-vis_cam_width/2,  vis_cam_height/2, wire_frame_depth, 1/vis_scale)) * vis_scale
    rb = inv_extrinsic @ np.array((vis_cam_width/2,   vis_cam_height/2, wire_frame_depth, 1/vis_scale)) * vis_scale
    
    p1.x, p1.y, p1.z = [x, lt[0]], [y, lt[1]],  [z, lt[2]]
    p2.x, p2.y, p2.z = [x, rt[0]], [y, rt[1]],  [z, rt[2]]
    p3.x, p3.y, p3.z = [x, lb[0]], [y, lb[1]], [z, lb[2]]
    p4.x, p4.y, p4.z = [x, rb[0]], [y, rb[1]], [z, rb[2]]
    p5.x, p5.y, p5.z = [lt[0], rt[0]], [lt[1], rt[1]], [lt[2], rt[2]]
    p6.x, p6.y, p6.z = [rt[0], rb[0]], [rt[1], rb[1]], [rt[2], rb[2]]
    p7.x, p7.y, p7.z = [rb[0], lb[0]], [rb[1], lb[1]], [rb[2], lb[2]]
    p8.x, p8.y, p8.z = [lb[0], lt[0]], [lb[1], lt[1]], [lb[2], lt[2]]
    p9.x, p9.y, p9.z = np.array([x]), np.array([y]), np.array([z])
    
    return [p1, p2, p3, p4, p5, p6, p7, p8, p9]

def dim(x):
    """ Determines the dimensionality of an array; 
        A helper function for update_picture. """
    d = 0
    el = x
    while True:
        try:
            el = el[0]
            d += 1
        except:
            break
    return d

def reshape(ar):
    """ Reshapes an array; A helper function for update_picture. """
    if dim(ar) == 3:
        return [k.reshape(-1) for k in ar]
    else:
        return ar.reshape(-1)

def update_picture(image, inv_extrinsic, vis_scale, old_plot):
    """ Updates the location of pictures within a wireframes. 
        This allows for animating the pictures. """
    image = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    image = cv2.resize(image, None, fx=0.1, fy=0.1) / 255
    img_height, img_width, _ = image.shape
    xx, yy = np.meshgrid(np.linspace(-vis_cam_width/2 * vis_scale,  vis_cam_width/2 * vis_scale,  img_width), 
                         np.linspace(-vis_cam_height/2 * vis_scale, vis_cam_height/2 * vis_scale, img_height))
    zz = np.ones_like(yy) * wire_frame_depth * vis_scale
    coords = np.stack([xx, yy, zz, np.ones_like(zz)]) 
    coords = coords.reshape(4, -1) 
     
    # Convert canera relative coordinates to world relative coordinates
    coords = inv_extrinsic @ coords
    old_color = old_plot.color.copy()
    xx, yy, zz, ones = coords.reshape(4, img_height, img_width) 
    
    x = reshape(xx)
    y = reshape(yy)
    z = reshape(zz)
    old_plot.x = x
    old_plot.y = y
    old_plot.z = z
    return old_plot
    
    
def project_points_to_picture(image, object_points, intrinsics, extrinsic):
    """ Perspective projects points to an image and draws them green. """
    image = image.copy()
    proj_matrix = intrinsics @ extrinsic
    object_points = proj_matrix @ object_points
    xs, ys, ones, disparity = object_points / object_points[2]
    
    for idx, (x, y) in enumerate(zip(xs, ys)):
        x = round(x)
        y = round(y)
        if (0 < y < image.shape[0] and
            0 < x < image.shape[1]):
            # Each point occupies a 20x20 pixel area in the image.
            image[y-10:y+10, x-10:x+10] = [0, 255, 0] 
    
    return image

def triangulate(p1, p2, p3, p4):
    """ Calculates the point triangulated by two lines. 
        Also returns that points projection onto line 1 and 2, 
        know as Pa and Pb in the math description in notebook 1. 
    """
    # Strip potential scale factor of homogenous coord
    p1 = p1[:3]
    p2 = p2[:3]
    p3 = p3[:3]
    p4 = p4[:3]
    
    p13 = p1 - p3
    p21 = p2 - p1
    p43 = p4 - p3
    
    d1321 = np.dot(p13, p21)
    d1343 = np.dot(p13, p43)
    d2121 = np.dot(p21, p21)
    d4321 = np.dot(p43, p21) 
    d4343 = np.dot(p43, p43) 
    
    mu_a = (d1343 * d4321 - d1321 * d4343) / (d2121 * d4343 - d4321 * d4321)
    mu_b = (d1343 + mu_a * d4321) / d4343
    
    point_on_line_1 = p1 + mu_a * p21
    point_on_line_2 = p3 + mu_b * p43
    
    adjoining_line = point_on_line_2 - point_on_line_1
    midpoint = adjoining_line / 2 
    
    triangulated_point = point_on_line_1 + midpoint
    
    return triangulated_point, point_on_line_1, point_on_line_2


def get_stereo_setup_with_correspondences():
    """ Returns all objects related to the stereo setup 
        presented at the end of notebook 1 for triangulating points.
    """
    images, intrinsics, distortions, rotation_vectors, translation_vectors, object_points = calibrate_cameras()
    camera_1_idx = 3
    camera_2_idx = 0 
    image_1 = images[camera_1_idx].copy()
    image_2 = images[camera_2_idx].copy()

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Find the keypoints and descriptors with SIFT
    keypoints_1, descriptors_1 = sift.detectAndCompute(image_1, None) # queryImage
    keypoints_2, descriptors_2 = sift.detectAndCompute(image_2, None) # trainimage

    # Match descriptors
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors_1, descriptors_2, k=2)

    # Apply ratio test 
    good = []
    for best_match, second_best_match in matches:
        if best_match.distance < 0.75 * second_best_match.distance:
            good.append([best_match])

    # Sort matches according to descriptor distance
    dists = [g[0].distance for g in good]
    good = list(sorted(zip(dists, good)))
    good = [list(g) for g in zip(*good)][1]

    # Select manually validated matches
    hand_picked_matches = [2, 9, 15, 16, 18, 19, 22, 23, 24, 27, 28, 29, 31, 34, 40, 41, 42]
    good = np.array(good, dtype=object)[hand_picked_matches]
    
    match_coords_1 = list()
    match_coords_2 = list()

    for i in good:
        i = i[0]
        keypoint_1 = keypoints_1[i.queryIdx]
        keypoint_2 = keypoints_2[i.trainIdx]
        keypoint_1_center = np.array(keypoint_1.pt)
        keypoint_2_center = np.array(keypoint_2.pt)
        x1, y1 = keypoint_1_center
        x2, y2 = keypoint_2_center
        match_coords_1.append([x1, y1, 1, 1])
        match_coords_2.append([x2, y2, 1, 1])
        color = (np.random.rand(3) * 255).astype(int).clip(50, 255).tolist()
        image1 = cv2.circle(image_1, keypoint_1_center.astype(int), 10, color, -1)
        image2 = cv2.circle(image_2, keypoint_2_center.astype(int), 10, color, -1)

    match_coords_1 = np.array(match_coords_1)
    match_coords_2 = np.array(match_coords_2)

    extrinsic_1 = extrinsics[camera_1_idx]
    extrinsic_2 = extrinsics[camera_2_idx]
    inv_extrinsic_1 = np.linalg.inv(extrinsic_1)
    inv_extrinsic_2 = np.linalg.inv(extrinsic_2)
    cam_center_1 = camera_centers[camera_1_idx]
    cam_center_2 = camera_centers[camera_2_idx]
    cam_x_1, cam_y_1, cam_z_1, _ = cam_center_1
    cam_x_2, cam_y_2, cam_z_2, _ = cam_center_2
    
    return [image_1, image_2], [extrinsic_1, extrinsic_2], [cam_center_1, cam_center_2], intrinsics, [match_coords_1, match_coords_2], object_points

def get_bunny():
    """Plots the Stanford bunny pointcloud and returns its points"""
    bunny_coords = np.load(open('data/bunny_point_cloud.npy', 'rb')) * 2
    b_xs, b_ys, b_zs = bunny_coords[:3]
    b_xs -= b_xs.mean()
    b_ys -= b_ys.mean()
    b_zs -= b_zs.mean()
    bunny_coords = np.array([b_xs, b_ys, b_zs])
    return bunny_coords

def random_angle(max_angle): 
    """ Returns a random angle in radians. """
    rad = np.radians(max_angle)
    rand_factor = np.random.uniform(low=-1, high=1)
    return rad * rand_factor
                                  
def random_rotation(max_angle=100): 
    """ Returns a matrix for random rotation around x, y, and z axis. """
    t_x = random_angle(max_angle)
    t_y = random_angle(max_angle) 
    t_z = random_angle(max_angle)
                                  
    r_x = np.array([[1,           0,            0], 
                    [0, np.cos(t_x), -np.sin(t_x)], 
                    [0, np.sin(t_x),  np.cos(t_x)]])
                                  
    r_y = np.array([[np.cos(t_y),  0, np.sin(t_y)], 
                    [0,            1,         0], 
                    [-np.sin(t_y), 0, np.cos(t_y)]])
                                  
    r_z = np.array([[np.cos(t_z), -np.sin(t_z), 0], 
                    [np.sin(t_z),  np.cos(t_z), 0], 
                    [0,            0,           1]])
    
    return r_x @ r_y @ r_z 

def random_translation(max_offset=10): 
    """ Returns a random translation vector. """
    return np.random.uniform(low=-max_offset, high=max_offset, size=3)

def distort_extrinsics(extrinsic, max_angle=100, max_trans=5): 
    """ Randomly distorts an extrinsic matrix such that 
        the pose it represents is rotated and moved.  
    """
    extrinsic = extrinsic.copy()
    rot = extrinsic[:3, :3]
    trans = extrinsic[3, :3]
    rand_rot = random_rotation(max_angle)
    rand_trans = random_translation(max_trans)
    extrinsic[:3, :3] = rand_rot @ extrinsic[:3, :3]
    extrinsic[:3, 3] = extrinsic[:3, 3] + rand_trans 
    return extrinsic
