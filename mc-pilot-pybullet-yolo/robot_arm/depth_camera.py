"""
Depth-camera utilities for front-of-base target perception in PyBullet.

The camera is mounted at the front of the robot base, looks toward the
throwing workspace, renders RGB-D images, and uses OpenCV color segmentation
to recover the target bin position in world coordinates.
"""

import cv2
import numpy as np
import pybullet as p


BIN_RADIUS = 0.075
BIN_HEIGHT = 0.18
BIN_WALL_THICKNESS = 0.008
BIN_FLOOR_THICKNESS = 0.01
BIN_NUM_WALLS = 12
BIN_RGBA = (0.18, 0.78, 0.28, 1.0)
BIN_HSV_LOWER = np.array([45, 70, 70], dtype=np.uint8)
BIN_HSV_UPPER = np.array([90, 255, 255], dtype=np.uint8)
BIN_CENTER_CORRECTION = 2.0 * BIN_RADIUS / np.pi


def _normalize(vec):
    vec = np.array(vec, dtype=float)
    norm = np.linalg.norm(vec)
    if norm < 1e-9:
        raise ValueError("Cannot normalize a near-zero vector")
    return vec / norm


def spawn_target_bin(
    client_id,
    target_xy,
    radius=BIN_RADIUS,
    height=BIN_HEIGHT,
    wall_thickness=BIN_WALL_THICKNESS,
    floor_thickness=BIN_FLOOR_THICKNESS,
    num_walls=BIN_NUM_WALLS,
    rgba=BIN_RGBA,
):
    """
    Spawn a hollow target bin centered at the requested ground-plane target.

    The collision geometry is a thin circular floor plus wall segments, so the
    ball can fall inside instead of colliding with a solid cylinder.

    Returns the PyBullet body id.
    """
    x, y = np.array(target_xy, dtype=float)
    floor_collision = p.createCollisionShape(
        p.GEOM_CYLINDER,
        radius=radius,
        height=floor_thickness,
        physicsClientId=client_id,
    )
    floor_visual = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=radius,
        length=floor_thickness,
        rgbaColor=list(rgba),
        physicsClientId=client_id,
    )

    inner_radius = max(0.01, radius - wall_thickness)
    wall_length = 2.0 * inner_radius * np.tan(np.pi / float(num_walls))
    wall_half_extents = [wall_thickness / 2.0, wall_length / 2.0, height / 2.0]
    wall_collision = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=wall_half_extents,
        physicsClientId=client_id,
    )
    wall_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=wall_half_extents,
        rgbaColor=list(rgba),
        physicsClientId=client_id,
    )

    link_masses = []
    link_collision_shape_indices = []
    link_visual_shape_indices = []
    link_positions = []
    link_orientations = []
    link_inertial_frame_positions = []
    link_inertial_frame_orientations = []
    link_parent_indices = []
    link_joint_types = []
    link_joint_axis = []

    wall_center_radius = inner_radius + (wall_thickness / 2.0)
    for wall_idx in range(int(num_walls)):
        theta = (2.0 * np.pi * wall_idx) / float(num_walls)
        wall_x = wall_center_radius * np.cos(theta)
        wall_y = wall_center_radius * np.sin(theta)
        wall_quat = p.getQuaternionFromEuler([0.0, 0.0, theta])

        link_masses.append(0.0)
        link_collision_shape_indices.append(wall_collision)
        link_visual_shape_indices.append(wall_visual)
        link_positions.append([wall_x, wall_y, height / 2.0])
        link_orientations.append(wall_quat)
        link_inertial_frame_positions.append([0.0, 0.0, 0.0])
        link_inertial_frame_orientations.append([0.0, 0.0, 0.0, 1.0])
        link_parent_indices.append(0)
        link_joint_types.append(p.JOINT_FIXED)
        link_joint_axis.append([0.0, 0.0, 0.0])

    return p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=floor_collision,
        baseVisualShapeIndex=floor_visual,
        basePosition=[float(x), float(y), floor_thickness / 2.0],
        linkMasses=link_masses,
        linkCollisionShapeIndices=link_collision_shape_indices,
        linkVisualShapeIndices=link_visual_shape_indices,
        linkPositions=link_positions,
        linkOrientations=link_orientations,
        linkInertialFramePositions=link_inertial_frame_positions,
        linkInertialFrameOrientations=link_inertial_frame_orientations,
        linkParentIndices=link_parent_indices,
        linkJointTypes=link_joint_types,
        linkJointAxis=link_joint_axis,
        physicsClientId=client_id,
    )


class ArmMountedDepthCamera:
    """
    Front-of-base RGB-D camera that estimates the target bin position.

    The camera is anchored to the robot base with a fixed front-facing offset
    and looks toward the nominal throwing workspace so the target region stays
    in frame from a stable viewpoint.
    """

    def __init__(
        self,
        client_id,
        arm_controller,
        width=320,
        height=240,
        fov_deg=95.0,
        near=0.02,
        far=3.0,
        local_offset=(0.16, 0.0, 0.62),
        workspace_focus=(0.9, 0.0, 0.08),
        renderer=None,
    ):
        self._cid = client_id
        self._arm = arm_controller
        self.width = int(width)
        self.height = int(height)
        self.fov_deg = float(fov_deg)
        self.near = float(near)
        self.far = float(far)
        self.local_offset = np.array(local_offset, dtype=float)
        self.workspace_focus = np.array(workspace_focus, dtype=float)
        self.renderer = p.ER_TINY_RENDERER if renderer is None else renderer

    def _camera_pose(self):
        base_pos, base_quat = p.getBasePositionAndOrientation(
            self._arm.arm_id,
            physicsClientId=self._cid,
        )
        rot = np.array(p.getMatrixFromQuaternion(base_quat)).reshape(3, 3)
        eye = np.array(base_pos, dtype=float) + rot.dot(self.local_offset)
        forward = _normalize(self.workspace_focus - eye)
        up = np.array([0.0, 0.0, 1.0])
        up = up - np.dot(up, forward) * forward
        if np.linalg.norm(up) < 1e-6:
            up = np.array([0.0, 0.0, 1.0])
        up = _normalize(up)
        target = eye + forward
        return eye, target, up

    def draw_debug_pose(self):
        """
        Draw a small world-space overlay showing where the camera is located
        and what direction it is facing.
        """
        eye, target, up = self._camera_pose()
        forward_tip = eye + 0.45 * _normalize(target - eye)
        up_tip = eye + 0.15 * up
        p.addUserDebugLine(
            eye.tolist(),
            forward_tip.tolist(),
            lineColorRGB=[0.1, 0.7, 1.0],
            lineWidth=3.0,
            physicsClientId=self._cid,
        )
        p.addUserDebugLine(
            eye.tolist(),
            up_tip.tolist(),
            lineColorRGB=[1.0, 1.0, 0.2],
            lineWidth=2.0,
            physicsClientId=self._cid,
        )
        p.addUserDebugText(
            "front base cam",
            eye.tolist(),
            textColorRGB=[0.9, 0.95, 1.0],
            textSize=1.2,
            physicsClientId=self._cid,
        )

    def capture(self):
        eye, target, up = self._camera_pose()
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=eye.tolist(),
            cameraTargetPosition=target.tolist(),
            cameraUpVector=up.tolist(),
        )
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.fov_deg,
            aspect=float(self.width) / float(self.height),
            nearVal=self.near,
            farVal=self.far,
        )
        _, _, rgba, depth, seg = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=self.renderer,
            physicsClientId=self._cid,
        )
        rgba = np.asarray(rgba, dtype=np.uint8).reshape(self.height, self.width, 4)
        depth = np.asarray(depth, dtype=np.float32).reshape(self.height, self.width)
        seg = np.asarray(seg).reshape(self.height, self.width)
        return {
            "rgba": rgba,
            "bgr": cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR),
            "depth": depth,
            "segmentation": seg,
            "view_matrix": np.array(view_matrix, dtype=np.float64),
            "projection_matrix": np.array(projection_matrix, dtype=np.float64),
            "eye": eye,
            "target": target,
            "up": up,
        }

    def detect_bin(self, observation, debug=False, projected_target_px=None, projected_target_visible=None):
        """
        Detect the colored bin in the RGB image and estimate its world-frame
        ground-plane center using the depth image.
        """
        bgr = observation["bgr"]
        depth = observation["depth"]

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, BIN_HSV_LOWER, BIN_HSV_UPPER)
        kernel = np.ones((5, 5), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise RuntimeError("OpenCV could not find the target bin in the front-base camera image")

        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < 40.0:
            raise RuntimeError("Detected target bin contour is too small to localize reliably")
        bbox = cv2.boundingRect(contour)

        contour_mask = np.zeros_like(mask)
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)
        ys, xs = np.where(contour_mask > 0)
        if xs.size == 0:
            raise RuntimeError("Target bin mask was empty after contour extraction")

        if xs.size > 3000:
            stride = int(np.ceil(xs.size / 3000.0))
            xs = xs[::stride]
            ys = ys[::stride]

        world_points = self._pixels_to_world(xs, ys, depth, observation)
        if world_points.size == 0:
            raise RuntimeError("Depth back-projection produced no valid target-bin points")

        surface_target_xy = np.median(world_points[:, :2], axis=0)
        camera_xy = np.array(observation["eye"][:2], dtype=float)
        ray_xy = surface_target_xy - camera_xy
        ray_norm = np.linalg.norm(ray_xy)
        if ray_norm > 1e-9:
            target_xy = surface_target_xy + BIN_CENTER_CORRECTION * (ray_xy / ray_norm)
        else:
            target_xy = surface_target_xy.copy()

        moments = cv2.moments(contour)
        if abs(moments["m00"]) > 1e-6:
            centroid_px = np.array(
                [
                    moments["m10"] / moments["m00"],
                    moments["m01"] / moments["m00"],
                ],
                dtype=float,
            )
        else:
            centroid_px = np.array([np.mean(xs), np.mean(ys)], dtype=float)

        result = {
            "target_xy": target_xy,
            "surface_target_xy": surface_target_xy,
            "mask": mask,
            "contour_mask": contour_mask,
            "contour": contour,
            "world_points": world_points,
            "centroid_px": centroid_px,
            "bbox": bbox,
        }

        if debug:
            result["debug_bgr"] = self.make_debug_image(
                observation=observation,
                contour=contour,
                centroid_px=centroid_px,
                target_xy=target_xy,
                bbox=bbox,
                projected_target_px=projected_target_px,
                projected_target_visible=projected_target_visible,
                surface_target_xy=surface_target_xy,
            )

        return result

    def capture_and_detect(self, debug=False, projected_target_px=None, projected_target_visible=None):
        observation = self.capture()
        detection = self.detect_bin(
            observation,
            debug=debug,
            projected_target_px=projected_target_px,
            projected_target_visible=projected_target_visible,
        )
        detection["observation"] = observation
        return detection

    def project_world_to_pixel(self, world_xyz, observation):
        """
        Project a world-frame point into the rendered image.
        Returns image pixel coordinates and a visibility flag.
        """
        view = observation["view_matrix"].reshape(4, 4, order="F")
        proj = observation["projection_matrix"].reshape(4, 4, order="F")
        world = np.array([world_xyz[0], world_xyz[1], world_xyz[2], 1.0], dtype=np.float64)
        clip = np.matmul(proj, np.matmul(view, world))
        if abs(clip[3]) < 1e-9:
            return {"pixel": None, "visible": False}

        ndc = clip[:3] / clip[3]
        pixel_x = ((ndc[0] + 1.0) * 0.5 * self.width) - 0.5
        pixel_y = ((1.0 - ndc[1]) * 0.5 * self.height) - 0.5
        visible = (
            clip[3] > 0.0
            and -1.0 <= ndc[0] <= 1.0
            and -1.0 <= ndc[1] <= 1.0
            and -1.0 <= ndc[2] <= 1.0
        )
        return {"pixel": np.array([pixel_x, pixel_y], dtype=float), "visible": bool(visible)}

    def _pixels_to_world(self, xs, ys, depth, observation):
        view = observation["view_matrix"].reshape(4, 4, order="F")
        proj = observation["projection_matrix"].reshape(4, 4, order="F")
        inv_vp = np.linalg.inv(np.matmul(proj, view))

        xs = np.asarray(xs, dtype=np.float64)
        ys = np.asarray(ys, dtype=np.float64)
        z_buf = depth[ys.astype(int), xs.astype(int)]
        valid = (z_buf > 1e-6) & (z_buf < 0.999999)
        if not np.any(valid):
            return np.empty((0, 3), dtype=np.float64)

        xs = xs[valid]
        ys = ys[valid]
        z_buf = z_buf[valid]

        x_ndc = (2.0 * (xs + 0.5) / float(self.width)) - 1.0
        y_ndc = 1.0 - (2.0 * (ys + 0.5) / float(self.height))
        z_ndc = (2.0 * z_buf) - 1.0
        clip = np.stack([x_ndc, y_ndc, z_ndc, np.ones_like(x_ndc)], axis=1)
        world_h = np.matmul(inv_vp, clip.T).T
        world = world_h[:, :3] / world_h[:, 3:4]

        keep = np.isfinite(world).all(axis=1) & (world[:, 2] > -0.05) & (world[:, 2] < 0.5)
        return world[keep]

    def make_debug_image(
        self,
        observation,
        contour=None,
        centroid_px=None,
        target_xy=None,
        bbox=None,
        projected_target_px=None,
        projected_target_visible=None,
        surface_target_xy=None,
        status_text=None,
    ):
        debug_bgr = observation["bgr"].copy()
        if contour is not None:
            cv2.drawContours(debug_bgr, [contour], -1, (0, 255, 255), 2)
        if bbox is not None:
            x, y, w, h = bbox
            cv2.rectangle(debug_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                debug_bgr,
                "detected bin",
                (x, max(18, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        if centroid_px is not None:
            center = tuple(np.round(centroid_px).astype(int).tolist())
            cv2.circle(debug_bgr, center, 4, (0, 0, 255), -1)
        if projected_target_px is not None:
            proj = tuple(np.round(projected_target_px).astype(int).tolist())
            colour = (255, 0, 0) if projected_target_visible else (60, 60, 255)
            cv2.drawMarker(
                debug_bgr,
                proj,
                colour,
                markerType=cv2.MARKER_CROSS,
                markerSize=14,
                thickness=2,
            )
            proj_label = "true bin in frame" if projected_target_visible else "true bin off frame"
            cv2.putText(
                debug_bgr,
                proj_label,
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                colour,
                2,
                cv2.LINE_AA,
            )
        if target_xy is not None:
            label = "target=({:.3f}, {:.3f})".format(float(target_xy[0]), float(target_xy[1]))
        else:
            label = "target=unavailable"
        cv2.putText(
            debug_bgr,
            label,
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        if surface_target_xy is not None:
            surf_label = "surface=({:.3f}, {:.3f})".format(
                float(surface_target_xy[0]),
                float(surface_target_xy[1]),
            )
            cv2.putText(
                debug_bgr,
                surf_label,
                (10, 76),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (180, 255, 180),
                2,
                cv2.LINE_AA,
            )
        if status_text is not None:
            cv2.putText(
                debug_bgr,
                status_text,
                (10, self.height - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (180, 220, 255),
                2,
                cv2.LINE_AA,
            )
        return debug_bgr
