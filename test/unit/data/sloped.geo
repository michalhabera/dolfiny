//+
Point(1) = {0, 0, 0, 1.0};
//+
Point(2) = {0, 1, 0, 1.0};
//+
Point(3) = {1, 1, 0, 1.0};
//+
Point(4) = {1, 0.5, 0, 1.0};
//+
Point(5) = {0.5, 0, 0, 1.0};
//+
Line(1) = {5, 1};
//+
Line(2) = {1, 2};
//+
Line(3) = {2, 3};
//+
Line(4) = {3, 4};
//+
Line(5) = {4, 5};
//+
Curve Loop(1) = {2, 3, 4, 5, 1};
//+
Plane Surface(1) = {1};
//+
Physical Curve("111") = {5};
//+
Physical Surface("100") = {1};
//+
Physical Curve("120") = {3};
//+
Physical Curve(4) = {2, 1, 4};
