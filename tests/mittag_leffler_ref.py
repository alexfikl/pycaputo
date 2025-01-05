# SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from pycaputo.typing import Array


class Reference(NamedTuple):
    alpha: float
    beta: float
    z: Array
    result: Array


MATHEMATICA_SINE_RESULTS = [
    Reference(
        alpha=0.8,
        beta=0.0,
        z=np.array([
            0.3973714758489759,
            4.337123067107832,
            9.436976599526211,
            11.966446164326065,
            4.228661446583491,
            8.254228971900226,
            9.606597156274805,
            4.02303689184242,
            9.291763618063065,
            12.053308486970948,
            4.634605786013896,
            10.596220126389728,
            1.8628980048456523,
            9.7361180213144,
            3.529228287596574,
            7.3256465055628865,
            10.58205511219592,
            1.803755989201992,
            9.78696612001362,
            9.157586259320016,
            6.216556621609033,
            12.008770491105345,
            4.7986831026021335,
            3.2855408504751935,
            0.050244620250545324,
            8.883907240688039,
            0.150798781551174,
            3.7098652208275666,
            0.8337651046293253,
            2.811106677425391,
            4.468110699999823,
            8.888241498516809,
            2.724258636010255,
            7.892049277161995,
            12.34460601093189,
            0.32483106786897054,
            3.5371126900621803,
            9.013412071666409,
            10.616426204356266,
            2.1388493281490017,
            9.544728115703922,
            12.367630447837552,
            11.679096370871942,
            12.180061300608862,
            10.927731404029867,
            5.1610772653008645,
            1.93552436486992,
            7.394601881444487,
            2.7754306713246972,
            2.1919042748251734,
            6.399174166078701,
            10.869311411799565,
            8.98694959384741,
            12.430899228811775,
            7.318634574790586,
            5.461516253473366,
            10.810829748976168,
            0.780066004800716,
            7.454472668453214,
            5.833514492415052,
            8.979202572653708,
            1.1333128353172857,
            7.854961085604028,
            1.1506962541591896,
        ]),
        result=np.array([
            0.8520289748778251,
            -0.646514331602799,
            -0.9576731642147873,
            0.608580461722061,
            -0.7267165142564863,
            -0.08964679136751934,
            -0.99408601377095,
            -0.8550534695011534,
            -0.9046692714144754,
            0.6750117329121876,
            -0.3913859647467542,
            -0.6568848675298108,
            -0.011811438901177468,
            -1.002761923426597,
            -1.0115238581220103,
            0.7418262824532021,
            -0.6675340798583855,
            0.04587481096428546,
            -1.0015881041321877,
            -0.8387933324773249,
            0.9224953127813448,
            0.6415565926657896,
            -0.23480160872705527,
            -1.001340293399999,
            0.5982462589710735,
            -0.6594308597685045,
            0.7396160694702447,
            -0.9810935149886109,
            0.7876860888510729,
            -0.8190267289321046,
            -0.5398139499750547,
            -0.6626925892645547,
            -0.7647693746984422,
            0.26864399270915434,
            0.8579633299388729,
            0.8352879733899692,
            -1.010867319677947,
            -0.7513072183138262,
            -0.6414672967655275,
            -0.2796250607813553,
            -0.9840623217702356,
            0.8694974488028159,
            0.3590016053599141,
            0.7626629230384231,
            -0.37504570056915504,
            0.1261881911850936,
            -0.08284357916681864,
            0.6942617352826553,
            -0.7974379917484039,
            -0.3297137403813819,
            0.9748241037010471,
            -0.4286091378839945,
            -0.7335045913660959,
            0.8987959378796063,
            0.746467890599342,
            0.4141319751303256,
            -0.48077248043110615,
            0.8080860316747164,
            0.6502604831660478,
            0.7156738359161521,
            -0.7281954703091954,
            0.6233208073197559,
            0.3041004658277862,
            0.6114521734892716,
        ]),
    ),
    Reference(
        alpha=4.391785201796967,
        beta=0.0,
        z=np.array([
            1.616371748043461,
            10.991013228464897,
            3.974447671252122,
            0.04302888376826175,
            0.08074172423101444,
            2.565077904743074,
            8.84884878292419,
            5.220511947439064,
            4.3333245907061375,
            12.260183576377099,
            3.904665336919148,
            5.897016147110271,
            1.9226239548645125,
            10.306523526930757,
            1.8067761470393844,
            0.7124042072476655,
            8.889973471785733,
            5.053423247901307,
            9.81109879517026,
            5.38062315543883,
            9.007215492384859,
            3.555506409745803,
            2.094802442089449,
            11.31940083168151,
            5.61146081873656,
            3.304825165308099,
            10.053544179615901,
            9.365826818212156,
            12.148913647793197,
            11.833856763086416,
            3.9278612316864034,
            1.0323168104977967,
            0.4651975708334195,
            4.347994756591003,
            0.20080529747470877,
            7.911478401562206,
            1.7551536401862116,
            5.210928241373821,
            6.23840072992029,
            9.648459830283837,
            8.9197170950804,
            3.283543831512075,
            7.206379784721452,
            11.210667942967035,
            12.487607673683371,
            0.46729767726112037,
            1.86592167336563,
            9.339727972842478,
            5.907194268864647,
            5.211791291854716,
            11.538137900173993,
            10.806502376602872,
            12.141062847027637,
            5.096880018264532,
            6.578140887859224,
            6.710850636168267,
            0.15085252888941092,
            11.563816367764858,
            12.457302364986237,
            10.90935044894837,
            1.561990852579374,
            2.9195248863716667,
            8.635855207384989,
            11.402761357501511,
        ]),
        result=np.array([
            0.7012375248645053,
            -0.8283996693472515,
            -1.0264800773721978,
            0.16492127488163533,
            0.24156907624183077,
            -0.09458798826502227,
            -0.05182030840143177,
            -0.45687414848744023,
            -1.0028367933366347,
            0.29634918494170576,
            -1.016262941434935,
            0.20632761104962383,
            0.4927919976412689,
            -1.0073644603479626,
            0.5795276241296959,
            0.8028780218328007,
            -0.09279234638415465,
            -0.6018098486263533,
            -0.8531681734249208,
            -0.3066936213243351,
            -0.2086219415927648,
            -0.8956711326892324,
            0.3494277722953084,
            -0.5992823710598271,
            -0.07854969750358391,
            -0.7444280440061728,
            -0.957548936727524,
            -0.5396348759469651,
            0.18859454345723195,
            -0.1251964544873393,
            -1.0201895802251009,
            0.8682720775823217,
            0.6661751550920147,
            -0.9991724582553647,
            0.41705024577225913,
            0.7677007550011674,
            0.6152430857071731,
            -0.46555104392033747,
            0.520693512324209,
            -0.7550433635810664,
            -0.12234517978944626,
            -0.7294315523105288,
            0.983316895680257,
            -0.6834985857857792,
            0.503483458630593,
            0.6676825998467386,
            0.5363329109623901,
            -0.5173388459758942,
            0.21627322122036605,
            -0.46477129474154444,
            -0.40984767319363213,
            -0.9199380350378437,
            0.1808840223852712,
            -0.5654979230600918,
            0.7715418364301804,
            0.8462321603812928,
            0.35193296790941986,
            -0.38617075240716026,
            0.4771818981123245,
            -0.8725499809795852,
            0.7306386177728572,
            -0.43167736584456406,
            0.15988904384796387,
            -0.5299415509753204,
        ]),
    ),
]
