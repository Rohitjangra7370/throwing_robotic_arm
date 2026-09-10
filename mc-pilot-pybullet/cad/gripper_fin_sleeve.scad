// ===================================================================
// Robotiq 2F-85 finger EXTENSION SLEEVE  (slip-on, clamped)
//
// Replaces the earlier bolt-on version, which needed the pad bolt pattern --
// dimensions this project has no authoritative source for. This one slides
// OVER the existing fingertip and pinches with a single cross bolt, so it
// needs no gripper disassembly, tolerates a wrong socket guess, and comes
// straight off again.
//
// PURPOSE (unchanged)
//   1. Put the grasp point 270 mm from the flange, matching
//      throw_pose_table_ext15.npy and results_kinetic_chain_gen3_ext15{,_narrow}/1.
//      run_hardware_throw.py REFUSES a --tool_offset_z that disagrees with the
//      table stamp, so the built number has to be right.
//   2. Break the encompassing grip. Measured 2026-09-09: the bare gripper cages
//      a 67 mm tennis ball until 8.33% closed, so the ball is not free until
//      ~175-250 ms after t_r and then drops onto the gripper. The tip is a
//      SHALLOW DISH -- contact below the ball's equator, so it lifts straight
//      out. Do not deepen it.
//   3. Grip force cannot be limited in software (GRIPPER_FORCE is rejected by
//      this firmware), so the section stays deep in the closing direction.
//
// SOCKET IS OVERSIZED ON PURPOSE. Measure the fingertip, set SOCKET_W/D to
// about +1.5 mm over it, and let the clamp take up the rest. Add a strip of
// rubber inside if it is still loose.
// ===================================================================

/* [Fit -- measure the fingertip, then add ~1.5 mm] */
SOCKET_W   = 24;    // across the finger
SOCKET_D   = 30;    // along the closing direction
SOCKET_L   = 70;    // LONG ON PURPOSE: engagement is adjustable, so the exact
                    // flange-to-fingertip measurement stops being critical.
                    // Slide it on further or less and clamp anywhere over a
                    // ~25 mm range, then MEASURE the resulting grasp point and
                    // set --tool_offset_z / re-derive the pose table to match.
WALL       = 2.6;
TIP_SOLID  = 34;    // solid block at the tip that carries the dish

/* [Length] */
DISH_FROM_FLANGE = 270;  // the number the checkpoint needs, mm
FLANGE_TO_TIP    = 120;  // nominal; the long socket absorbs the error
EXTENSION  = DISH_FROM_FLANGE - FLANGE_TO_TIP;   // sleeve reach past the tip

/* [Clamp] */
SLIT_W     = 2.5;
CLAMP_D    = 4.5;   // M4 clearance
NUT_AF     = 7.2;   // M4 nut across flats
NUT_T      = 3.4;

/* [Tip] */
BALL_D     = 67;
DISH_DEPTH = 4;
DISH_R     = 35;
DISH_L     = 26;

$fn = 64;
OW = SOCKET_W + 2*WALL;
OD = SOCKET_D + 2*WALL;
TOTAL = SOCKET_L + EXTENSION;
DISH_Z = TOTAL - 18;

// STRAIGHT-SIDED TUBE. The first attempt tapered the body via hull(), which
// made the cross-bolt and slit exit through a sloped face and produced a
// non-manifold solid. Straight walls also keep it a real tube: a solid block
// this size would be ~260 g per finger, which the wrist does not need.
module body() {
    translate([-OW/2, -OD/2, 0]) cube([OW, OD, TOTAL]);
}

// Cavity runs the WHOLE length except a solid tip block that carries the dish.
module cavity() {
    translate([-SOCKET_W/2, -SOCKET_D/2, -1])
        cube([SOCKET_W, SOCKET_D, TOTAL - TIP_SOLID + 1]);
}

module clamp() {
    // slit through the outer (-Y) wall so the socket can close on the finger
    translate([-SLIT_W/2, -OD/2 - 1, -1])
        cube([SLIT_W, WALL + 2, SOCKET_L - 6]);
    // TWO cross bolts + captive nuts -- one is not enough to hold a 70 mm
    // socket square against grip load.
    for (z = [SOCKET_L - 16, SOCKET_L - 48])
        translate([0, -(SOCKET_D/2 + WALL/2), z]) rotate([0, 90, 0]) {
            cylinder(d = CLAMP_D, h = OW + 4, center = true);
            translate([0, 0, OW/2 - NUT_T])
                cylinder(d = NUT_AF/cos(30), h = NUT_T + 2, $fn = 6);
        }
}

module dish() {
    intersection() {
        translate([0, OD/2 + DISH_R - DISH_DEPTH, DISH_Z]) sphere(r = DISH_R);
        translate([-OW, OD/2 - DISH_DEPTH, DISH_Z - DISH_L/2])
            cube([2*OW, DISH_DEPTH + 2, DISH_L]);
    }
}

// MIRROR=0 -> right-hand part, MIRROR=1 -> left-hand. The two fingers oppose
// each other, so the dish must face inward on both while the clamp slit stays
// outward -- that is a mirror pair, not one part rotated.
MIRROR = 0;
module sleeve() { difference() { body(); cavity(); clamp(); dish(); } }
if (MIRROR) mirror([0,1,0]) sleeve(); else sleeve();
