// ===================================================================
// Robotiq 2F-85 finger EXTENSION FIN
// for the Kinova Gen3 throwing rig -- "fins" as in MC-PILOT (Turcato et al.)
//
// WHY
//   1. Move the grasp point 0.12 m -> 0.27 m from the flange (+150 mm), to
//      match throw_pose_table_ext15.npy and the checkpoints
//      results_kinetic_chain_gen3_ext15{,_narrow}/1. Those are trained for a
//      0.27 m TCP and run_hardware_throw.py REFUSES a mismatched
//      --tool_offset_z, so the built length has to match the number.
//
//   2. Break the encompassing grip. MEASURED 2026-09-09: the bare 2F-85 holds
//      a 67 mm tennis ball in a deep cage -- still caged at 8.33% closed, so
//      the ball is not free until ~175-250 ms after t_r, by which point the
//      palm is vertical (z_tool . up = 0.999) and the ball drops back onto the
//      gripper instead of flying. Hence the TIP IS A SHALLOW DISH, not a cup:
//      all contact stays BELOW the ball's equator, so it lifts straight out
//      the instant the fins part. Deepening this dish would undo the point of
//      the part.
//
//   3. Grip force cannot be softened in software -- GRIPPER_FORCE mode is
//      rejected by this arm's firmware ("Force command mode not supported",
//      measured 2026-09-09). So the beam has to take full grip load: it is
//      DEEP in the closing direction (Y) and ribbed. Do not flatten it.
//
// >>> UNVERIFIED: the pad mount below is a PLACEHOLDER. <<<
//   The only 2F-85 geometry on disk is primitive collision geometry from an
//   uncertain generic-gripper source, explicitly not authoritative. MEASURE
//   the real pad -- bolt spacing, bolt size, footprint -- and set PAD_*.
// ===================================================================

/* [Extension] */
FIN_LENGTH   = 150;    // mm. MUST equal (tool_offset_z - 0.12 m) in mm.

/* [Beam] */
BEAM_W       = 14;     // across the finger
BEAM_D_ROOT  = 34;     // depth along the CLOSING direction -- the bending axis
BEAM_D_TIP   = 24;

/* [Pad mount -- PLACEHOLDER, MEASURE FIRST] */
PAD_W        = 22;
PAD_H        = 37.5;
PAD_T        = 6;
BOLT_SPACING = 20;
BOLT_D       = 3.4;    // M3 clearance
BOLT_HEAD_D  = 6.2;
BOLT_HEAD_H  = 3.2;

/* [Ball cradle] */
BALL_D       = 67;     // tennis; a 40 mm TT ball also seats in this dish
DISH_DEPTH   = 4;      // SHALLOW BY DESIGN -- see note 2 above
DISH_R       = 35;     // slightly proud of the ball radius (33.5)
DISH_L       = 26;     // pocket length -- bond the TPU/rubber facing in here

/* [Stiffening] */
RIB_T        = 3;
RIB_N        = 4;

DISH_Z = PAD_T + FIN_LENGTH - 20;   // dish centre height

$fn = 64;

module mount_plate() {
    difference() {
        translate([-PAD_W/2, -PAD_H/2, 0]) cube([PAD_W, PAD_H, PAD_T]);
        for (s = [-1, 1])
            translate([0, s*BOLT_SPACING/2, -0.1]) {
                cylinder(d = BOLT_D, h = PAD_T + 0.2);
                translate([0, 0, PAD_T - BOLT_HEAD_H])
                    cylinder(d = BOLT_HEAD_D, h = BOLT_HEAD_H + 0.2);
            }
    }
}

module beam() {
    hull() {
        translate([-BEAM_W/2, -BEAM_D_ROOT/2, PAD_T])
            cube([BEAM_W, BEAM_D_ROOT, 1]);
        translate([-BEAM_W/2, -BEAM_D_TIP/2, PAD_T + FIN_LENGTH - 1])
            cube([BEAM_W, BEAM_D_TIP, 1]);
    }
}

// Stiffening MUST NOT touch the +Y gripping face. A full collar puts rib
// edges where the ball sits, which is the encompassing-cage geometry this
// part exists to remove -- caught in the first render. Gussets run on the
// OUTER (-Y) face and the two side faces only, and they deepen the section in
// Y, which is the axis grip force bends about.
module gussets() {
    for (i = [0 : RIB_N - 1]) {
        f  = i / max(RIB_N - 1, 1);
        z  = PAD_T + 10 + f * (FIN_LENGTH - 60);
        d  = BEAM_D_ROOT + (BEAM_D_TIP - BEAM_D_ROOT) * (z - PAD_T)/FIN_LENGTH;
        h  = RIB_T * (2.2 - 1.2*f);            // taller near the root
        // outer face only: spans -d/2-RIB_T .. +d/2 (stops short of +Y face)
        translate([-BEAM_W/2 - RIB_T, -d/2 - RIB_T, z])
            cube([BEAM_W + 2*RIB_T, d/2 + RIB_T, h]);
    }
}

// Root gusset: peak bending moment is at the mount, taper it out.
module root_gusset() {
    hull() {
        translate([-BEAM_W/2, -BEAM_D_ROOT/2 - 5, PAD_T])
            cube([BEAM_W, 5, 0.5]);
        translate([-BEAM_W/2, -BEAM_D_ROOT/2 - 0.5, PAD_T + 42])
            cube([BEAM_W, 0.5, 0.5]);
    }
}

// Gripping face is +Y (toward the opposing fin).
//
// The dish must be a LOCALISED pocket. A ball-radius sphere cutting a face
// that stands 12 mm proud scoops 53 mm of beam (caught in the second render),
// which both weakens the section and re-wraps the ball. Intersecting with a
// box confines it to DISH_L, keeping the ball-matching curvature without the
// long scoop. Bond the compliant facing straight into this pocket -- a
// separate rectangular recess just overlapped it and added nothing.
module tip_features() {
    intersection() {
        translate([0, BEAM_D_TIP/2 + DISH_R - DISH_DEPTH, DISH_Z])
            sphere(r = DISH_R);
        translate([-BEAM_W, BEAM_D_TIP/2 - DISH_DEPTH, DISH_Z - DISH_L/2])
            cube([2*BEAM_W, DISH_DEPTH + 2, DISH_L]);
    }
}

module fin() {
    difference() {
        union() { mount_plate(); beam(); gussets(); root_gusset(); }
        tip_features();
    }
}

fin();
