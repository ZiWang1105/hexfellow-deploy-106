"""Measured ChArUco board detection with explicit camera intrinsics."""
import cv2
import numpy as np


def make_board(config):
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, config['dictionary']))
    board = cv2.aruco.CharucoBoard((config['squares_x'], config['squares_y']),
        config['square_length_m'], config['marker_length_m'], dictionary,
        np.asarray(config['marker_ids'], dtype=np.int32))
    board.setLegacyPattern(config.get('legacy_pattern', False))
    return board


def detect(image, config, k, distortion, *, pixel_to_pinhole=None):
    """Reject a bad vision frame without terminating the camera stream."""
    try:
        return _detect(image, config, k, distortion, pixel_to_pinhole=pixel_to_pinhole)
    except cv2.error as error:
        return {'markers': 0, 'corners': 0, 'accepted': False,
                'reason': f'ChArUco processing failed for this frame: {error}'}, image.copy()


def _draw(result, operation, *args):
    # Overlays are diagnostic; a drawing failure must not change pose validity.
    try:
        operation(*args)
    except cv2.error as error:
        result.setdefault('overlay_warnings', []).append(str(error))


def _detect(image, config, k, distortion, *, pixel_to_pinhole=None):
    board = make_board(config)
    params = cv2.aruco.CharucoParameters()
    matrix = np.asarray(k, dtype=float)
    coefficients = np.asarray(distortion, dtype=float)
    # Inverse-Brown SDK coefficients cannot be used as OpenCV forward-Brown
    # coefficients. Interpolate raw corners by local homography, then map their
    # rays through the SDK for PnP in the pinhole pixel coordinate system.
    if pixel_to_pinhole is None:
        params.cameraMatrix = matrix
        params.distCoeffs = coefficients
    detector = cv2.aruco.CharucoDetector(board, params)
    corners, ids, markers, marker_ids = detector.detectBoard(image)
    annotated = image.copy()
    result = {'markers': 0 if marker_ids is None else np.asarray(marker_ids).size,
              'corners': 0 if ids is None else np.asarray(ids).size, 'accepted': False}
    if marker_ids is not None:
        _draw(result, cv2.aruco.drawDetectedMarkers, annotated, markers, marker_ids)
    if ids is None or result['corners'] < 8:
        result['reason'] = 'Need at least 8 ChArUco corners'; return result, annotated
    # OpenCV 4 returns (N,1,2)/(N,1), while 5 returns (N,2)/(N).
    # Drawing still requires a two-channel point array in both versions.
    ids = np.ascontiguousarray(ids, dtype=np.int32).reshape(-1)
    points = np.asarray(corners) if corners is not None else np.empty(0)
    all_obj = np.asarray(board.getChessboardCorners()).reshape(-1, 3)
    if (points.size != 2*ids.size or not np.all(np.isfinite(points))
            or len(np.unique(ids)) != ids.size or np.any(ids < 0) or np.any(ids >= len(all_obj))):
        result['reason'] = 'Invalid ChArUco corner/ID correspondence'; return result, annotated
    corners = np.ascontiguousarray(points, dtype=np.float32).reshape(-1, 1, 2)
    _draw(result, cv2.aruco.drawDetectedCornersCharuco, annotated, corners, ids.reshape(-1, 1))
    raw_corners = corners.reshape(-1, 2).copy()
    if pixel_to_pinhole is not None:
        mapped = np.asarray(pixel_to_pinhole(raw_corners), dtype=np.float32)
        if mapped.shape != raw_corners.shape or not np.isfinite(mapped).all():
            result['reason'] = 'Invalid SDK corner-ray rectification'; return result, annotated
        corners = np.ascontiguousarray(mapped).reshape(-1, 1, 2)
        coefficients = np.zeros(5)
    obj = all_obj[ids]
    if np.linalg.matrix_rank(obj[:, :2]-obj[:, :2].mean(axis=0), tol=1e-6) < 2:
        result['reason'] = 'Collinear corners'; return result, annotated
    ok, rvec, tvec = cv2.solvePnP(obj, corners, matrix, coefficients, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        result['reason'] = 'PnP failed'; return result, annotated
    rvec, tvec = cv2.solvePnPRefineLM(obj, corners, matrix, coefficients, rvec, tvec)
    projected,_ = cv2.projectPoints(obj,rvec,tvec,matrix,coefficients)
    # Normalize both outputs to prevent accidental N-by-N NumPy broadcasting.
    rms = float(np.sqrt(np.mean(np.sum((projected.reshape(-1,2)-corners.reshape(-1,2))**2,axis=-1))))
    rotation = cv2.Rodrigues(rvec)[0]
    camera_points = obj @ rotation.T + tvec.reshape(3)
    t = np.eye(4); t[:3,:3]=rotation; t[:3,3]=tvec.ravel()
    result.update(T_camera_from_board=t.tolist(), corner_ids=ids.ravel().tolist(),
        image_points=corners.reshape(-1,2).tolist(), object_points=obj.tolist(),
        reprojection_rms_px=rms, distance_m=float(np.linalg.norm(tvec)),
        accepted=bool(rms < 1.0 and np.all(camera_points[:,2] > 0)))
    result['reason'] = 'visual quality passed; robot synchronization still required' if result['accepted'] else 'Reprojection/depth check failed'
    if pixel_to_pinhole is not None:
        result.update(raw_image_points=raw_corners.tolist(),
            pixel_coordinates='SDK-deprojected pinhole pixels; raw_image_points refer to saved native RGB',
            pnp_distortion=[0.]*5)
    else:
        _draw(result, cv2.drawFrameAxes, annotated,matrix,coefficients,rvec,tvec,.04)
    return result, annotated
