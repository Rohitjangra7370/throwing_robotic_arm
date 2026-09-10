use <gripper_fin.scad>
// Two fins facing each other with a tennis ball pinched at the tips.
// Fin separation is set so the 67 mm ball is held by the dishes.
BALL_D = 67; FIN_LENGTH = 150; PAD_T = 6; BEAM_D_TIP = 24; DISH_DEPTH = 4;
SEP = BALL_D - 2*DISH_DEPTH + 2*(BEAM_D_TIP/2);   // face-to-face + beam halves
module fin_() { import("gripper_fin.stl"); }
translate([0, -SEP/2, 0]) fin_();
translate([0,  SEP/2, 0]) mirror([0,1,0]) fin_();
color([0.85,0.75,0.2,0.85])
    translate([0, 0, PAD_T + FIN_LENGTH - 18]) sphere(d = BALL_D, $fn=64);
