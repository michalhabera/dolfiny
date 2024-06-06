#!/usr/bin/env python3

from mpi4py import MPI


def mesh_schwedler_gmshapi(
    name="schwedler",
    order=1,
    msh_file=None,
    comm=MPI.COMM_WORLD,
):
    """
    Create mesh of Schwedler truss using the Python API of Gmsh.
    """

    tdim = 1  # target topological dimension

    # Perform Gmsh work only on rank = 0

    if comm.rank == 0:
        import gmsh

        # Initialise gmsh and set options
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)

        # Add model under given name
        gmsh.model.add(name)

        # Create points and lines
        points = [
            gmsh.model.geo.addPoint(22.90, 0.00, 0.00),  # p0
            gmsh.model.geo.addPoint(22.12, 5.93, 0.00),  # p1
            gmsh.model.geo.addPoint(19.83, 11.45, 0.00),  # p2
            gmsh.model.geo.addPoint(16.19, 16.19, 0.00),  # p3
            gmsh.model.geo.addPoint(11.45, 19.83, 0.00),  # p4
            gmsh.model.geo.addPoint(5.93, 22.12, 0.00),  # p5
            gmsh.model.geo.addPoint(0.00, 22.90, 0.00),  # p6
            gmsh.model.geo.addPoint(-5.93, 22.12, 0.00),  # p7
            gmsh.model.geo.addPoint(-11.45, 19.83, 0.00),  # p8
            gmsh.model.geo.addPoint(-16.19, 16.19, 0.00),  # p9
            gmsh.model.geo.addPoint(-19.83, 11.45, 0.00),  # p10
            gmsh.model.geo.addPoint(-22.12, 5.93, 0.00),  # p11
            gmsh.model.geo.addPoint(-22.90, 0.00, 0.00),  # p12
            gmsh.model.geo.addPoint(-22.12, -5.93, 0.00),  # p13
            gmsh.model.geo.addPoint(-19.83, -11.45, 0.00),  # p14
            gmsh.model.geo.addPoint(-16.19, -16.19, 0.00),  # p15
            gmsh.model.geo.addPoint(-11.45, -19.83, 0.00),  # p16
            gmsh.model.geo.addPoint(-5.93, -22.12, 0.00),  # p17
            gmsh.model.geo.addPoint(0.00, -22.90, 0.00),  # p18
            gmsh.model.geo.addPoint(5.93, -22.12, 0.00),  # p19
            gmsh.model.geo.addPoint(11.45, -19.83, 0.00),  # p20
            gmsh.model.geo.addPoint(16.19, -16.19, 0.00),  # p21
            gmsh.model.geo.addPoint(19.83, -11.45, 0.00),  # p22
            gmsh.model.geo.addPoint(22.12, -5.93, 0.00),  # p23
            gmsh.model.geo.addPoint(21.37, 0.00, 1.79),
            gmsh.model.geo.addPoint(20.64, 5.53, 1.79),
            gmsh.model.geo.addPoint(18.51, 10.69, 1.79),
            gmsh.model.geo.addPoint(15.11, 15.11, 1.79),
            gmsh.model.geo.addPoint(10.69, 18.51, 1.79),
            gmsh.model.geo.addPoint(5.53, 20.64, 1.79),
            gmsh.model.geo.addPoint(0.00, 21.37, 1.79),
            gmsh.model.geo.addPoint(-5.53, 20.64, 1.79),
            gmsh.model.geo.addPoint(-10.69, 18.51, 1.79),
            gmsh.model.geo.addPoint(-15.11, 15.11, 1.79),
            gmsh.model.geo.addPoint(-18.51, 10.69, 1.79),
            gmsh.model.geo.addPoint(-20.64, 5.53, 1.79),
            gmsh.model.geo.addPoint(-21.37, 0.00, 1.79),
            gmsh.model.geo.addPoint(-20.64, -5.53, 1.79),
            gmsh.model.geo.addPoint(-18.51, -10.69, 1.79),
            gmsh.model.geo.addPoint(-15.11, -15.11, 1.79),
            gmsh.model.geo.addPoint(-10.69, -18.51, 1.79),
            gmsh.model.geo.addPoint(-5.53, -20.64, 1.79),
            gmsh.model.geo.addPoint(0.00, -21.37, 1.79),
            gmsh.model.geo.addPoint(5.53, -20.64, 1.79),
            gmsh.model.geo.addPoint(10.69, -18.51, 1.79),
            gmsh.model.geo.addPoint(15.11, -15.11, 1.79),
            gmsh.model.geo.addPoint(18.51, -10.69, 1.79),
            gmsh.model.geo.addPoint(20.64, -5.53, 1.79),
            gmsh.model.geo.addPoint(16.41, 0.00, 3.26),  # p48
            gmsh.model.geo.addPoint(15.85, 4.25, 3.26),  # p49
            gmsh.model.geo.addPoint(14.21, 8.21, 3.26),  # p50
            gmsh.model.geo.addPoint(11.60, 11.60, 3.26),  # p51
            gmsh.model.geo.addPoint(8.21, 14.21, 3.26),  # p52
            gmsh.model.geo.addPoint(4.25, 15.85, 3.26),  # p53
            gmsh.model.geo.addPoint(0.00, 16.41, 3.26),  # p54
            gmsh.model.geo.addPoint(-4.25, 15.85, 3.26),  # p55
            gmsh.model.geo.addPoint(-8.21, 14.21, 3.26),  # p56
            gmsh.model.geo.addPoint(-11.60, 11.60, 3.26),  # p57
            gmsh.model.geo.addPoint(-14.21, 8.21, 3.26),  # p58
            gmsh.model.geo.addPoint(-15.85, 4.25, 3.26),  # p59
            gmsh.model.geo.addPoint(-16.41, 0.00, 3.26),  # p60
            gmsh.model.geo.addPoint(-15.85, -4.25, 3.26),  # p61
            gmsh.model.geo.addPoint(-14.21, -8.21, 3.26),  # p62
            gmsh.model.geo.addPoint(-11.60, -11.60, 3.26),  # p63
            gmsh.model.geo.addPoint(-8.21, -14.21, 3.26),  # p64
            gmsh.model.geo.addPoint(-4.25, -15.85, 3.26),  # p65
            gmsh.model.geo.addPoint(0.00, -16.41, 3.26),  # p66
            gmsh.model.geo.addPoint(4.25, -15.85, 3.26),  # p67
            gmsh.model.geo.addPoint(8.21, -14.21, 3.26),  # p68
            gmsh.model.geo.addPoint(11.60, -11.60, 3.26),  # p69
            gmsh.model.geo.addPoint(14.21, -8.21, 3.26),  # p70
            gmsh.model.geo.addPoint(15.85, -4.25, 3.26),  # p71
            gmsh.model.geo.addPoint(8.78, 0.00, 4.27),  # p72
            gmsh.model.geo.addPoint(8.48, 2.27, 4.27),  # p73
            gmsh.model.geo.addPoint(7.60, 4.39, 4.27),  # p74
            gmsh.model.geo.addPoint(6.21, 6.21, 4.27),  # p75
            gmsh.model.geo.addPoint(4.39, 7.60, 4.27),  # p76
            gmsh.model.geo.addPoint(2.27, 8.48, 4.27),  # p77
            gmsh.model.geo.addPoint(0.00, 8.78, 4.27),  # p78
            gmsh.model.geo.addPoint(-2.27, 8.48, 4.27),  # p79
            gmsh.model.geo.addPoint(-4.39, 7.60, 4.27),  # p80
            gmsh.model.geo.addPoint(-6.21, 6.21, 4.27),  # p81
            gmsh.model.geo.addPoint(-7.60, 4.39, 4.27),  # p82
            gmsh.model.geo.addPoint(-8.48, 2.27, 4.27),  # p83
            gmsh.model.geo.addPoint(-8.78, 0.00, 4.27),  # p84
            gmsh.model.geo.addPoint(-8.48, -2.27, 4.27),  # p85
            gmsh.model.geo.addPoint(-7.60, -4.39, 4.27),  # p86
            gmsh.model.geo.addPoint(-6.21, -6.21, 4.27),  # p87
            gmsh.model.geo.addPoint(-4.39, -7.60, 4.27),  # p88
            gmsh.model.geo.addPoint(-2.27, -8.48, 4.27),  # p89
            gmsh.model.geo.addPoint(0.00, -8.78, 4.27),  # p90
            gmsh.model.geo.addPoint(2.27, -8.48, 4.27),  # p91
            gmsh.model.geo.addPoint(4.39, -7.60, 4.27),  # p92
            gmsh.model.geo.addPoint(6.21, -6.21, 4.27),  # p93
            gmsh.model.geo.addPoint(7.60, -4.39, 4.27),  # p94
            gmsh.model.geo.addPoint(8.48, -2.27, 4.27),  # p95
            gmsh.model.geo.addPoint(0.00, 0.00, 4.58),  # p96
            #
            gmsh.model.geo.addPoint(0.00, 0.00, 4.58 + 0.00001),  # p97  ext
        ]
        lines = [
            gmsh.model.geo.addLine(points[96], points[72]),  # l0
            gmsh.model.geo.addLine(points[96], points[73]),  # l1
            gmsh.model.geo.addLine(points[96], points[74]),  # l2
            gmsh.model.geo.addLine(points[96], points[75]),  # l3
            gmsh.model.geo.addLine(points[96], points[76]),  # l4
            gmsh.model.geo.addLine(points[96], points[77]),  # l5
            gmsh.model.geo.addLine(points[96], points[78]),  # l6
            gmsh.model.geo.addLine(points[96], points[79]),  # l7
            gmsh.model.geo.addLine(points[96], points[80]),  # l8
            gmsh.model.geo.addLine(points[96], points[81]),  # l9
            gmsh.model.geo.addLine(points[96], points[82]),  # l10
            gmsh.model.geo.addLine(points[96], points[83]),  # l11
            gmsh.model.geo.addLine(points[96], points[84]),  # l12
            gmsh.model.geo.addLine(points[96], points[85]),  # l13
            gmsh.model.geo.addLine(points[96], points[86]),  # l14
            gmsh.model.geo.addLine(points[96], points[87]),  # l15
            gmsh.model.geo.addLine(points[96], points[88]),  # l16
            gmsh.model.geo.addLine(points[96], points[89]),  # l17
            gmsh.model.geo.addLine(points[96], points[90]),  # l18
            gmsh.model.geo.addLine(points[96], points[91]),  # l19
            gmsh.model.geo.addLine(points[96], points[92]),  # l20
            gmsh.model.geo.addLine(points[96], points[93]),  # l21
            gmsh.model.geo.addLine(points[96], points[94]),  # l22
            gmsh.model.geo.addLine(points[96], points[95]),  # l23
            gmsh.model.geo.addLine(points[72], points[48]),  # l24
            gmsh.model.geo.addLine(points[73], points[49]),  # l25
            gmsh.model.geo.addLine(points[74], points[50]),  # l26
            gmsh.model.geo.addLine(points[75], points[51]),  # l27
            gmsh.model.geo.addLine(points[76], points[52]),  # l28
            gmsh.model.geo.addLine(points[77], points[53]),  # l29
            gmsh.model.geo.addLine(points[78], points[54]),  # l30
            gmsh.model.geo.addLine(points[79], points[55]),  # l31
            gmsh.model.geo.addLine(points[80], points[56]),  # l32
            gmsh.model.geo.addLine(points[81], points[57]),  # l33
            gmsh.model.geo.addLine(points[82], points[58]),  # l34
            gmsh.model.geo.addLine(points[83], points[59]),  # l35
            gmsh.model.geo.addLine(points[84], points[60]),  # l36
            gmsh.model.geo.addLine(points[85], points[61]),  # l37
            gmsh.model.geo.addLine(points[86], points[62]),  # l38
            gmsh.model.geo.addLine(points[87], points[63]),  # l39
            gmsh.model.geo.addLine(points[88], points[64]),  # l40
            gmsh.model.geo.addLine(points[89], points[65]),  # l41
            gmsh.model.geo.addLine(points[90], points[66]),  # l42
            gmsh.model.geo.addLine(points[91], points[67]),  # l43
            gmsh.model.geo.addLine(points[92], points[68]),  # l44
            gmsh.model.geo.addLine(points[93], points[69]),  # l45
            gmsh.model.geo.addLine(points[94], points[70]),  # l46
            gmsh.model.geo.addLine(points[95], points[71]),  # l47
            gmsh.model.geo.addLine(points[48], points[24]),  # l48
            gmsh.model.geo.addLine(points[49], points[25]),  # l49
            gmsh.model.geo.addLine(points[50], points[26]),  # l50
            gmsh.model.geo.addLine(points[51], points[27]),  # l51
            gmsh.model.geo.addLine(points[52], points[28]),  # l52
            gmsh.model.geo.addLine(points[53], points[29]),  # l53
            gmsh.model.geo.addLine(points[54], points[30]),  # l54
            gmsh.model.geo.addLine(points[55], points[31]),  # l55
            gmsh.model.geo.addLine(points[56], points[32]),  # l56
            gmsh.model.geo.addLine(points[57], points[33]),  # l57
            gmsh.model.geo.addLine(points[58], points[34]),  # l58
            gmsh.model.geo.addLine(points[59], points[35]),  # l59
            gmsh.model.geo.addLine(points[60], points[36]),  # l60
            gmsh.model.geo.addLine(points[61], points[37]),  # l61
            gmsh.model.geo.addLine(points[62], points[38]),  # l62
            gmsh.model.geo.addLine(points[63], points[39]),  # l63
            gmsh.model.geo.addLine(points[64], points[40]),  # l64
            gmsh.model.geo.addLine(points[65], points[41]),  # l65
            gmsh.model.geo.addLine(points[66], points[42]),  # l66
            gmsh.model.geo.addLine(points[67], points[43]),  # l67
            gmsh.model.geo.addLine(points[68], points[44]),  # l68
            gmsh.model.geo.addLine(points[69], points[45]),  # l69
            gmsh.model.geo.addLine(points[70], points[46]),  # l70
            gmsh.model.geo.addLine(points[71], points[47]),  # l71
            gmsh.model.geo.addLine(points[24], points[0]),  # l72
            gmsh.model.geo.addLine(points[25], points[1]),  # l73
            gmsh.model.geo.addLine(points[26], points[2]),  # l74
            gmsh.model.geo.addLine(points[27], points[3]),  # l75
            gmsh.model.geo.addLine(points[28], points[4]),  # l76
            gmsh.model.geo.addLine(points[29], points[5]),  # l77
            gmsh.model.geo.addLine(points[30], points[6]),  # l78
            gmsh.model.geo.addLine(points[31], points[7]),  # l79
            gmsh.model.geo.addLine(points[32], points[8]),  # l80
            gmsh.model.geo.addLine(points[33], points[9]),  # l81
            gmsh.model.geo.addLine(points[34], points[10]),  # l82
            gmsh.model.geo.addLine(points[35], points[11]),  # l83
            gmsh.model.geo.addLine(points[36], points[12]),  # l84
            gmsh.model.geo.addLine(points[37], points[13]),  # l85
            gmsh.model.geo.addLine(points[38], points[14]),  # l86
            gmsh.model.geo.addLine(points[39], points[15]),  # l87
            gmsh.model.geo.addLine(points[40], points[16]),  # l88
            gmsh.model.geo.addLine(points[41], points[17]),  # l89
            gmsh.model.geo.addLine(points[42], points[18]),  # l90
            gmsh.model.geo.addLine(points[43], points[19]),  # l91
            gmsh.model.geo.addLine(points[44], points[20]),  # l92
            gmsh.model.geo.addLine(points[45], points[21]),  # l93
            gmsh.model.geo.addLine(points[46], points[22]),  # l94
            gmsh.model.geo.addLine(points[47], points[23]),  # l95
            gmsh.model.geo.addLine(points[0], points[1]),  # l96
            gmsh.model.geo.addLine(points[1], points[2]),  # l97
            gmsh.model.geo.addLine(points[2], points[3]),  # l98
            gmsh.model.geo.addLine(points[3], points[4]),  # l99
            gmsh.model.geo.addLine(points[4], points[5]),  # l100
            gmsh.model.geo.addLine(points[5], points[6]),  # l101
            gmsh.model.geo.addLine(points[6], points[7]),  # l102
            gmsh.model.geo.addLine(points[7], points[8]),  # l103
            gmsh.model.geo.addLine(points[8], points[9]),  # l104
            gmsh.model.geo.addLine(points[9], points[10]),  # l105
            gmsh.model.geo.addLine(points[10], points[11]),  # l106
            gmsh.model.geo.addLine(points[11], points[12]),  # l107
            gmsh.model.geo.addLine(points[12], points[13]),  # l108
            gmsh.model.geo.addLine(points[13], points[14]),  # l109
            gmsh.model.geo.addLine(points[14], points[15]),  # l110
            gmsh.model.geo.addLine(points[15], points[16]),  # l111
            gmsh.model.geo.addLine(points[16], points[17]),  # l112
            gmsh.model.geo.addLine(points[17], points[18]),  # l113
            gmsh.model.geo.addLine(points[18], points[19]),  # l114
            gmsh.model.geo.addLine(points[19], points[20]),  # l115
            gmsh.model.geo.addLine(points[20], points[21]),  # l116
            gmsh.model.geo.addLine(points[21], points[22]),  # l117
            gmsh.model.geo.addLine(points[22], points[23]),  # l118
            gmsh.model.geo.addLine(points[23], points[0]),  # l119
            gmsh.model.geo.addLine(points[24], points[25]),  # l120
            gmsh.model.geo.addLine(points[25], points[26]),  # l121
            gmsh.model.geo.addLine(points[26], points[27]),  # l122
            gmsh.model.geo.addLine(points[27], points[28]),  # l123
            gmsh.model.geo.addLine(points[28], points[29]),  # l124
            gmsh.model.geo.addLine(points[29], points[30]),  # l125
            gmsh.model.geo.addLine(points[30], points[31]),  # l126
            gmsh.model.geo.addLine(points[31], points[32]),  # l127
            gmsh.model.geo.addLine(points[32], points[33]),  # l128
            gmsh.model.geo.addLine(points[33], points[34]),  # l129
            gmsh.model.geo.addLine(points[34], points[35]),  # l130
            gmsh.model.geo.addLine(points[35], points[36]),  # l131
            gmsh.model.geo.addLine(points[36], points[37]),  # l132
            gmsh.model.geo.addLine(points[37], points[38]),  # l133
            gmsh.model.geo.addLine(points[38], points[39]),  # l134
            gmsh.model.geo.addLine(points[39], points[40]),  # l135
            gmsh.model.geo.addLine(points[40], points[41]),  # l136
            gmsh.model.geo.addLine(points[41], points[42]),  # l137
            gmsh.model.geo.addLine(points[42], points[43]),  # l138
            gmsh.model.geo.addLine(points[43], points[44]),  # l139
            gmsh.model.geo.addLine(points[44], points[45]),  # l140
            gmsh.model.geo.addLine(points[45], points[46]),  # l141
            gmsh.model.geo.addLine(points[46], points[47]),  # l142
            gmsh.model.geo.addLine(points[47], points[24]),  # l143
            gmsh.model.geo.addLine(points[48], points[49]),  # l144
            gmsh.model.geo.addLine(points[49], points[50]),  # l145
            gmsh.model.geo.addLine(points[50], points[51]),  # l146
            gmsh.model.geo.addLine(points[51], points[52]),  # l147
            gmsh.model.geo.addLine(points[52], points[53]),  # l148
            gmsh.model.geo.addLine(points[53], points[54]),  # l149
            gmsh.model.geo.addLine(points[54], points[55]),  # l150
            gmsh.model.geo.addLine(points[55], points[56]),  # l151
            gmsh.model.geo.addLine(points[56], points[57]),  # l152
            gmsh.model.geo.addLine(points[57], points[58]),  # l153
            gmsh.model.geo.addLine(points[58], points[59]),  # l154
            gmsh.model.geo.addLine(points[59], points[60]),  # l155
            gmsh.model.geo.addLine(points[60], points[61]),  # l156
            gmsh.model.geo.addLine(points[61], points[62]),  # l157
            gmsh.model.geo.addLine(points[62], points[63]),  # l158
            gmsh.model.geo.addLine(points[63], points[64]),  # l159
            gmsh.model.geo.addLine(points[64], points[65]),  # l160
            gmsh.model.geo.addLine(points[65], points[66]),  # l161
            gmsh.model.geo.addLine(points[66], points[67]),  # l162
            gmsh.model.geo.addLine(points[67], points[68]),  # l163
            gmsh.model.geo.addLine(points[68], points[69]),  # l164
            gmsh.model.geo.addLine(points[69], points[70]),  # l165
            gmsh.model.geo.addLine(points[70], points[71]),  # l166
            gmsh.model.geo.addLine(points[71], points[48]),  # l167
            gmsh.model.geo.addLine(points[72], points[73]),  # l168
            gmsh.model.geo.addLine(points[73], points[74]),  # l169
            gmsh.model.geo.addLine(points[74], points[75]),  # l170
            gmsh.model.geo.addLine(points[75], points[76]),  # l171
            gmsh.model.geo.addLine(points[76], points[77]),  # l172
            gmsh.model.geo.addLine(points[77], points[78]),  # l173
            gmsh.model.geo.addLine(points[78], points[79]),  # l174
            gmsh.model.geo.addLine(points[79], points[80]),  # l175
            gmsh.model.geo.addLine(points[80], points[81]),  # l176
            gmsh.model.geo.addLine(points[81], points[82]),  # l177
            gmsh.model.geo.addLine(points[82], points[83]),  # l178
            gmsh.model.geo.addLine(points[83], points[84]),  # l179
            gmsh.model.geo.addLine(points[84], points[85]),  # l180
            gmsh.model.geo.addLine(points[85], points[86]),  # l181
            gmsh.model.geo.addLine(points[86], points[87]),  # l182
            gmsh.model.geo.addLine(points[87], points[88]),  # l183
            gmsh.model.geo.addLine(points[88], points[89]),  # l184
            gmsh.model.geo.addLine(points[89], points[90]),  # l185
            gmsh.model.geo.addLine(points[90], points[91]),  # l186
            gmsh.model.geo.addLine(points[91], points[92]),  # l187
            gmsh.model.geo.addLine(points[92], points[93]),  # l188
            gmsh.model.geo.addLine(points[93], points[94]),  # l189
            gmsh.model.geo.addLine(points[94], points[95]),  # l190
            gmsh.model.geo.addLine(points[95], points[72]),  # l191
            gmsh.model.geo.addLine(points[0], points[25]),  # l192
            gmsh.model.geo.addLine(points[1], points[26]),  # l193
            gmsh.model.geo.addLine(points[2], points[27]),  # l194
            gmsh.model.geo.addLine(points[3], points[28]),  # l195
            gmsh.model.geo.addLine(points[4], points[29]),  # l196
            gmsh.model.geo.addLine(points[5], points[30]),  # l197
            gmsh.model.geo.addLine(points[6], points[31]),  # l198
            gmsh.model.geo.addLine(points[7], points[32]),  # l199
            gmsh.model.geo.addLine(points[8], points[33]),  # l200
            gmsh.model.geo.addLine(points[9], points[34]),  # l201
            gmsh.model.geo.addLine(points[10], points[35]),  # l202
            gmsh.model.geo.addLine(points[11], points[36]),  # l203
            gmsh.model.geo.addLine(points[12], points[37]),  # l204
            gmsh.model.geo.addLine(points[13], points[38]),  # l205
            gmsh.model.geo.addLine(points[14], points[39]),  # l206
            gmsh.model.geo.addLine(points[15], points[40]),  # l207
            gmsh.model.geo.addLine(points[16], points[41]),  # l208
            gmsh.model.geo.addLine(points[17], points[42]),  # l209
            gmsh.model.geo.addLine(points[18], points[43]),  # l210
            gmsh.model.geo.addLine(points[19], points[44]),  # l211
            gmsh.model.geo.addLine(points[20], points[45]),  # l212
            gmsh.model.geo.addLine(points[21], points[46]),  # l213
            gmsh.model.geo.addLine(points[22], points[47]),  # l214
            gmsh.model.geo.addLine(points[23], points[24]),  # l215
            gmsh.model.geo.addLine(points[24], points[49]),  # 216
            gmsh.model.geo.addLine(points[25], points[50]),  # 217
            gmsh.model.geo.addLine(points[26], points[51]),  # 218
            gmsh.model.geo.addLine(points[27], points[52]),  # 219
            gmsh.model.geo.addLine(points[28], points[53]),  # 220
            gmsh.model.geo.addLine(points[29], points[54]),  # 221
            gmsh.model.geo.addLine(points[30], points[55]),  # 222
            gmsh.model.geo.addLine(points[31], points[56]),  # 223
            gmsh.model.geo.addLine(points[32], points[57]),  # 224
            gmsh.model.geo.addLine(points[33], points[58]),  # 225
            gmsh.model.geo.addLine(points[34], points[59]),  # 226
            gmsh.model.geo.addLine(points[35], points[60]),  # 227
            gmsh.model.geo.addLine(points[36], points[61]),  # 228
            gmsh.model.geo.addLine(points[37], points[62]),  # 229
            gmsh.model.geo.addLine(points[38], points[63]),  # 230
            gmsh.model.geo.addLine(points[39], points[64]),  # 231
            gmsh.model.geo.addLine(points[40], points[65]),  # 232
            gmsh.model.geo.addLine(points[41], points[66]),  # 233
            gmsh.model.geo.addLine(points[42], points[67]),  # 234
            gmsh.model.geo.addLine(points[43], points[68]),  # 235
            gmsh.model.geo.addLine(points[44], points[69]),  # 236
            gmsh.model.geo.addLine(points[45], points[70]),  # 237
            gmsh.model.geo.addLine(points[46], points[71]),  # 238
            gmsh.model.geo.addLine(points[47], points[48]),  # 239
            gmsh.model.geo.addLine(points[48], points[73]),  # l240
            gmsh.model.geo.addLine(points[49], points[74]),  # l241
            gmsh.model.geo.addLine(points[50], points[75]),  # l242
            gmsh.model.geo.addLine(points[51], points[76]),  # l243
            gmsh.model.geo.addLine(points[52], points[77]),  # l244
            gmsh.model.geo.addLine(points[53], points[78]),  # l245
            gmsh.model.geo.addLine(points[54], points[79]),  # l246
            gmsh.model.geo.addLine(points[55], points[80]),  # l247
            gmsh.model.geo.addLine(points[56], points[81]),  # l248
            gmsh.model.geo.addLine(points[57], points[82]),  # l249
            gmsh.model.geo.addLine(points[58], points[83]),  # l250
            gmsh.model.geo.addLine(points[59], points[84]),  # l251
            gmsh.model.geo.addLine(points[60], points[85]),  # l252
            gmsh.model.geo.addLine(points[61], points[86]),  # l253
            gmsh.model.geo.addLine(points[62], points[87]),  # l254
            gmsh.model.geo.addLine(points[63], points[88]),  # l255
            gmsh.model.geo.addLine(points[64], points[89]),  # l256
            gmsh.model.geo.addLine(points[65], points[90]),  # l257
            gmsh.model.geo.addLine(points[66], points[91]),  # l258
            gmsh.model.geo.addLine(points[67], points[92]),  # l259
            gmsh.model.geo.addLine(points[68], points[93]),  # l260
            gmsh.model.geo.addLine(points[69], points[94]),  # l261
            gmsh.model.geo.addLine(points[70], points[95]),  # l262
            gmsh.model.geo.addLine(points[71], points[72]),  # l263
            #
            gmsh.model.geo.addLine(points[96], points[97]),  # l264
        ]

        # Sync
        gmsh.model.geo.synchronize()
        # Define physical groups for subdomains (! target tag > 0)
        domain = 0
        gmsh.model.addPhysicalGroup(tdim, lines, domain)
        gmsh.model.setPhysicalName(tdim, domain, "domain")
        # Define physical groups for interfaces (! target tag > 0)
        support = 1
        gmsh.model.addPhysicalGroup(tdim - 1, points[0:24], support)
        gmsh.model.setPhysicalName(tdim - 1, support, "support")
        connect = 2
        gmsh.model.addPhysicalGroup(tdim - 1, points[72:73], connect)
        gmsh.model.setPhysicalName(tdim - 1, connect, "connect")
        verytop = 3
        gmsh.model.addPhysicalGroup(tdim - 1, points[96:97], verytop)
        gmsh.model.setPhysicalName(tdim - 1, verytop, "verytop")
        exploit = 4
        gmsh.model.addPhysicalGroup(tdim - 1, points[97:98], exploit)
        gmsh.model.setPhysicalName(tdim - 1, exploit, "exploit")

        # Set refinement along curve direction
        for line in lines:
            gmsh.model.mesh.setTransfiniteCurve(line, numNodes=2, meshType="Progression", coef=1.0)

        # Generate the mesh
        gmsh.model.mesh.generate()

        # Set geometric order of mesh cells
        gmsh.model.mesh.setOrder(order)

        # Optional: Write msh file
        if msh_file is not None:
            gmsh.write(msh_file)

    return gmsh.model if comm.rank == 0 else None, tdim


if __name__ == "__main__":
    mesh_schwedler_gmshapi(msh_file="schwedler.msh")
