# SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from pycaputo.utils import Array


class Reference(NamedTuple):
    alpha: float
    beta: float
    z: Array
    result: Array


# Mathematica code
#
#   Alpha = RandomReal[{0, 2}]
#   Beta = RandomReal[{0, 2}]
#   n = 64;
#   r = Sqrt[RandomReal[{0, 1}, n]];
#   Theta = RandomReal[{0, 2 Pi}, n];
#   z = r Exp[I Theta];
#   StringReplace[ExportString[z, "PythonExpression"], {"Complex" -> "complex"}]
#   mlE = MittagLefflerE[Alpha, Beta, z];
#   StringReplace[ExportString[mlE, "PythonExpression"], {"Complex" -> "complex"}]
#
# The export isn't super useful, but can be transformed into the below with
# minimal work.

MATHEMATICA_RESULTS = [
    Reference(
        alpha=0.174899,
        beta=1.67099,
        z=np.array([
            complex(0.1628902913674126, 0.2083623595842386),
            complex(0.301536708156155, 0.21062059397081628),
            complex(0.39424588531030047, -0.7494425417160024),
            complex(0.7382149419446931, -0.4636461602089559),
            complex(-0.6489663108914042, 0.04344246073893843),
            complex(0.04547882065188447, -0.5013553429442241),
            complex(0.14646419633703633, 0.003802000553102663),
            complex(0.7912464197148865, 0.29279490021800586),
            complex(-0.7960064342437211, 0.2696416428762632),
            complex(-0.26523627529556476, -0.4031114751534427),
            complex(-0.8810706512426298, -0.36701813102091063),
            complex(0.8094133437980116, 0.1304373024559951),
            complex(0.5153958375839096, -0.6084342632889432),
            complex(-0.07050294563464991, 0.9476723488726135),
            complex(0.4269673797577663, -0.517692502602484),
            complex(-0.8956118740315082, -0.03394166549934992),
            complex(0.07948381275355267, 0.10658046077898364),
            complex(0.12591736822216223, 0.15170597004144376),
            complex(-0.07541226650436991, -0.35053337462755546),
            complex(-0.9093446164806414, -0.2616824820367234),
            complex(-0.4036374637698667, -0.3516754829538942),
            complex(-0.3307437820932125, 0.9006038725622076),
            complex(0.23436199208940497, 0.4342665688859537),
            complex(-0.11198786170283688, 0.6995047442467573),
            complex(-0.5358224586950843, 0.5214576883857608),
            complex(0.4226147794060986, 0.7793633056055025),
            complex(0.6878619269241517, 0.5186163052910183),
            complex(0.5472086215007289, 0.010987295057402797),
            complex(-0.18047918805778831, 0.851490157925599),
            complex(0.9104293079840747, -0.08941652096532575),
            complex(-0.3470357828475446, 0.9249848432073775),
            complex(-0.12852583575592852, -0.7160593943815391),
            complex(0.2675678748588535, 0.18973455083555016),
            complex(-0.06520248438471239, 0.9321765967710463),
            complex(-0.13689894598567437, -0.20415890457819053),
            complex(-0.656237485599001, 0.4490840714264489),
            complex(-0.4457617230437835, -0.21836609885170746),
            complex(0.8623397382217126, 0.016130544900451276),
            complex(-0.5850271278306072, -0.787410014634918),
            complex(0.4480045251771779, -0.19410391637990299),
            complex(0.5919342607806333, 0.037018370558882296),
            complex(0.22626325815581141, 0.9015550742671155),
            complex(0.5007890025716334, 0.6224880653343845),
            complex(-0.6997351111337514, -0.10106453173350705),
            complex(-0.3036342641127915, -0.9281039149963616),
            complex(-0.5805491932034624, -0.48648194869301975),
            complex(0.9551177836833279, 0.27172891762367946),
            complex(0.38859998637589743, 0.06153099353231953),
            complex(-0.6875385501315435, -0.08402098656432701),
            complex(-0.2113737410866143, -0.6673815303566756),
            complex(0.4261497534879802, 0.23243133396317597),
            complex(0.28192418863846935, 0.6694521577917383),
            complex(-0.22780462125647258, 0.09072783832185019),
            complex(-0.8649127697342129, 0.06748432819214654),
            complex(-0.3045816877891596, 0.1278262278238243),
            complex(0.7602295647372045, 0.45439767268827497),
            complex(0.39628405976909153, 0.06351705346939075),
            complex(0.32016599577073457, -0.1579894330842515),
            complex(0.5704319597833866, -0.6699244630767638),
            complex(0.32783746474479425, -0.41085666537631393),
            complex(0.20876521109743873, 0.43296289841277036),
            complex(-0.6144315688587789, 0.01753612389861867),
            complex(-0.3522615566387325, -0.2593356101828267),
            complex(-0.23219796959192296, -0.7044707801581663),
        ]),
        result=np.array([
            complex(1.2437275859163386, 0.29181736736623537),
            complex(1.4428911841201872, 0.4014254046723001),
            complex(0.7661833388633713, -0.8994824074723301),
            complex(1.2423115129677644, -1.8033485962436593),
            complex(0.6801285632719222, 0.017634585022908947),
            complex(0.9266105616417191, -0.4679590442396093),
            complex(1.286476876753461, 0.005399010185786296),
            complex(2.2151692445896303, 2.173717407107352),
            complex(0.6122952475303798, 0.09074902267021563),
            complex(0.8054865390387981, -0.2502113587936722),
            complex(0.5764033420609205, -0.11118793230826808),
            complex(3.6949172349819253, 1.5511822681645466),
            complex(0.959269284258511, -1.108170323085209),
            complex(0.5978039683797056, 0.5179373767518174),
            complex(1.1185352728828026, -0.932805265879189),
            complex(0.5929718087370248, -0.010493707028687696),
            complex(1.1838019118837952, 0.13021493238814674),
            complex(1.2258023237467275, 0.2012099254739996),
            complex(0.9409364801254111, -0.2960960051724321),
            complex(0.5783958764971714, -0.07838249146571792),
            complex(0.7524055296347694, -0.18457049744604587),
            complex(0.5845411696605753, 0.38906656374577686),
            complex(1.1160543353747057, 0.5968016434814373),
            complex(0.7305302154024049, 0.4477094813056163),
            complex(0.6573233324600328, 0.21952692549319316),
            complex(0.723918011410709, 0.9276112010364499),
            complex(1.0972770152320332, 1.569159663033654),
            complex(2.2726413161765895, 0.04643357341048015),
            complex(0.6331302859900843, 0.4472798542508498),
            complex(5.473517261482698, -2.0524558095560037),
            complex(0.5722300293762856, 0.38670006269138096),
            complex(0.716375602527143, -0.44326141929834306),
            complex(1.4056934324407855, 0.33779009191405934),
            complex(0.6068110541118296, 0.5194532139874899),
            complex(0.9502445025526367, -0.165013159297337),
            complex(0.6328831580043264, 0.1691185938762645),
            complex(0.7579826237145304, -0.11214408809489977),
            complex(5.0771133201159415, 0.2931794661401849),
            complex(0.5716928899392714, -0.28020192134579475),
            complex(1.7540120081840762, -0.54770396012462),
            complex(2.4658843901328598, 0.18315562339102703),
            complex(0.6342869684675078, 0.7152972337911103),
            complex(0.9361207053908563, 1.078801513340877),
            complex(0.6583217797406878, -0.03856487114194557),
            complex(0.5778260705708518, -0.4045335134455752),
            complex(0.6504641282976656, -0.19706477621432997),
            complex(2.2887657712850826, 4.160956137501392),
            complex(1.7362405658126627, 0.15711762043259797),
            complex(0.6636814418922726, -0.03254762188600388),
            complex(0.7161344163350002, -0.3854412558475292),
            complex(1.6451910046162788, 0.5975518406671285),
            complex(0.8608541274024621, 0.7639844096506241),
            complex(0.9035669812243269, 0.0648313357972673),
            complex(0.602039997383656, 0.02152267180205876),
            complex(0.8485000606425763, 0.08101076387043742),
            complex(1.2535361880394564, 1.9076994791459752),
            complex(1.754872375833724, 0.16564972491771776),
            complex(1.5227780448255028, -0.3245877469129779),
            complex(0.8255705325188497, -1.1858657771613508),
            complex(1.22647511062436, -0.6966830033464962),
            complex(1.0971586295137816, 0.5676349810162057),
            complex(0.6948230352294312, 0.007422152300618182),
            complex(0.7990139574141036, -0.14968385264174128),
            complex(0.6920942543944105, -0.3870531012043887),
        ]),
    ),
    Reference(
        alpha=0.751203,
        beta=0.122446,
        z=np.array([
            complex(0.4802878499284181, -0.0032236682474701974),
            complex(-0.9064362434769684, -0.10200682328498324),
            complex(-0.7256429354959082, 0.3083444898906649),
            complex(-0.35985602741681294, -0.6337710134392535),
            complex(0.45026619103145116, 0.29632153087280216),
            complex(-0.33369139449073043, 0.5010116610175944),
            complex(-0.2639233266012419, 0.8352890815281259),
            complex(0.2261967749351338, -0.8630031309346597),
            complex(0.48015631524236485, -0.7418565900006923),
            complex(-0.09737860243830153, 0.00005784848778595554),
            complex(-0.8494683056784619, 0.43896744031111534),
            complex(0.2174325626539914, 0.4368118666133587),
            complex(0.49800613458971094, -0.22245884571023716),
            complex(-0.7446063607165465, 0.6011880338439931),
            complex(0.007597165343116019, 0.15360157089998125),
            complex(-0.8712602940581669, -0.33458252679416195),
            complex(-0.9096932669327341, -0.27364027840509225),
            complex(-0.41295477432153355, -0.44427043031892754),
            complex(0.3019673635660516, 0.8764389183760192),
            complex(-0.5042700802004172, 0.5414968484594962),
            complex(0.010093593767620751, 0.7702859690006619),
            complex(-0.3661751062931662, 0.5489827746308057),
            complex(0.5006437179311577, 0.179522289095254),
            complex(0.24756778941946161, 0.7760489626898581),
            complex(-0.031293665742987205, -0.8484748017832647),
            complex(0.2720879467703164, 0.5739368069327991),
            complex(-0.6892226620400904, -0.6103595273235535),
            complex(-0.37394642803056666, 0.5231610906255074),
            complex(0.4599147724473998, -0.3451514285310666),
            complex(0.5531943083300768, 0.26160884300521847),
            complex(0.3361759066951848, -0.6430255918701958),
            complex(-0.3224766709105355, -0.6373209388308299),
            complex(-0.7906198162216603, 0.3593079903104605),
            complex(0.03305722954940247, 0.03583347966984273),
            complex(-0.24100025527675079, -0.3204722504414717),
            complex(-0.496371063058441, -0.5533209285533341),
            complex(-0.1314171180683846, 0.23159572218399613),
            complex(0.8171943592493863, 0.5405501683675972),
            complex(-0.0971770366172918, -0.25973112832183515),
            complex(0.24002836257918247, -0.23865150021369574),
            complex(0.5032746185809676, -0.04114318971686787),
            complex(0.26120609272108153, 0.9150883293113866),
            complex(-0.3943602443158216, 0.8094916991436153),
            complex(0.09346098408679829, 0.034938148791374996),
            complex(-0.14975491483520834, -0.14000255442021736),
            complex(-0.04510829926410517, -0.9745828867539839),
            complex(0.1273035510430993, -0.1906050689467695),
            complex(-0.6015930751179782, 0.4976359785114223),
            complex(0.12724897462554643, 0.7959274516039915),
            complex(-0.49293203070735603, 0.03815794442600685),
            complex(0.22854270910390362, -0.35954061902086437),
            complex(-0.11362104972510859, 0.4144843607571114),
            complex(0.4290740149412757, 0.4765477916267551),
            complex(0.753296817756454, 0.4446207901185475),
            complex(-0.38815413278556016, -0.5618616905318383),
            complex(-0.012810384301562578, 0.7248623587689025),
            complex(0.9428105045573836, 0.08308045259654132),
            complex(-0.041716449484714216, 0.2222839432176228),
            complex(-0.120891442053256, 0.925121236382459),
            complex(0.41907774866942504, 0.061714753558041166),
            complex(0.6077971711389362, -0.1545481777631072),
            complex(-0.1906107631283034, -0.09939436474266024),
            complex(0.36487002836534793, -0.7497655162044318),
            complex(-0.36005387504528624, -0.4190022573477348),
        ]),
        result=np.array([
            complex(0.9475975621930657, -0.009075313472949442),
            complex(-0.18380917149807985, -0.004905115750937758),
            complex(-0.18793249835095227, 0.03003930439366273),
            complex(-0.25828553182048575, -0.1430346768715853),
            complex(0.612400074432649, 0.7236688106849234),
            complex(-0.19809788273839207, 0.148301711957196),
            complex(-0.3728079152630782, 0.16438719244130665),
            complex(-0.6015130320263382, -0.6363117691358832),
            complex(-0.4569743225184076, -1.256381065315986),
            complex(0.0504810437422722, 0.00004172050139845629),
            complex(-0.20775011941193428, 0.01869279958344235),
            complex(0.07376831172683442, 0.5675545758967282),
            complex(0.8374516680344387, -0.6256050435898778),
            complex(-0.23439827123274581, 0.030611536943270793),
            complex(0.11041076937338702, 0.14044014474409575),
            complex(-0.19682214160935732, -0.0154341403175228),
            complex(-0.19265262787085877, -0.010937903485570133),
            complex(-0.1863245522918113, -0.11088509027772095),
            complex(-0.6779939750173426, 0.7583700227720149),
            complex(-0.2243611138818704, 0.08873478872070172),
            complex(-0.39238210485808933, 0.39181705022230023),
            complex(-0.22144653959359012, 0.13869707026279168),
            complex(0.9000419140441712, 0.5152854900408195),
            complex(-0.465718741913062, 0.7142794626864553),
            complex(-0.4560816451806099, -0.3269479196418946),
            complex(-0.10324200925961502, 0.7446476891562284),
            complex(-0.23883269669312604, -0.04082524017064405),
            complex(-0.21134995088315786, 0.13368699530529005),
            complex(0.5446782888071929, -0.8387766468342642),
            complex(0.9213570857101017, 0.8204444314219848),
            complex(-0.21349222716484464, -0.8937198323955937),
            complex(-0.26076007134217205, -0.16048267066411012),
            complex(-0.19733642225231737, 0.024618704590593028),
            complex(0.15990460783078847, 0.035549920692606846),
            complex(-0.10088484114063256, -0.14455004269335048),
            complex(-0.22785223087197634, -0.09141713206463753),
            complex(-0.01688089241950287, 0.14581274697864505),
            complex(0.6343771770123678, 2.5235992238401526),
            complex(-0.008504080163088987, -0.17599041763686313),
            complex(0.3220138810315031, -0.3685260031319445),
            complex(1.0083787838667932, -0.12185290020058022),
            complex(-0.7126817744473577, 0.6546109845378769),
            complex(-0.3308404040223465, 0.10821795715582526),
            complex(0.22434329510075843, 0.04007205121318517),
            complex(-0.0005076644005321908, -0.08689060157810104),
            complex(-0.5570992375340619, -0.25162617020322603),
            complex(0.21365957456758386, -0.22978702150754177),
            complex(-0.2150878803958618, 0.061602723570495704),
            complex(-0.4557758801359368, 0.5230483789045755),
            complex(-0.1280311849507452, 0.0094262107449988),
            complex(0.18271668766317242, -0.5070925417961618),
            complex(-0.10024604040114633, 0.2433125417117068),
            complex(0.21971685461046597, 0.9808864385994625),
            complex(0.9502180175555998, 1.9571603336324481),
            complex(-0.22784429507700052, -0.13027448844441705),
            complex(-0.34208228946042024, 0.3766248401465044),
            complex(3.1757774721617094, 0.6553903220896002),
            complex(0.044233808548391165, 0.17590048146609394),
            complex(-0.4795990181788108, 0.2217521119030066),
            complex(0.7761356659915831, 0.15067954901994113),
            complex(1.2638305128488256, -0.5689730065087247),
            complex(-0.016651933212456088, -0.05607855325898438),
            complex(-0.44806854729812323, -0.9566560520207098),
            complex(-0.16939210938354946, -0.12550439027939184),
        ]),
    ),
    Reference(
        alpha=0.417448,
        beta=0.345816,
        z=np.array([
            complex(-0.7405547773732496, 1.3931978536891505),
            complex(2.065525170266034, 0.5812396100208923),
            complex(1.2666625773737774, -1.67003823969862),
            complex(1.1157260090435983, -0.2941376913526355),
            complex(-0.764089926903913, -0.326105508734025),
            complex(0.4440565473235797, 2.055281956395518),
            complex(-0.7592100561256875, -2.0683882712137946),
            complex(-1.4765651025001978, -0.8307896579650716),
            complex(1.598371724111996, 1.5345276466017896),
            complex(-0.530796398535269, 2.055619263914736),
            complex(-0.8613487695832428, -1.3799302778622247),
            complex(0.9323346883166552, 0.19736994315385742),
            complex(-0.14611579669509656, 2.1700547893904667),
            complex(-0.5766243512674717, 1.2215770049993302),
            complex(-1.8466990101448353, 0.6389718556983008),
            complex(0.31858230174202107, 0.7887130137014613),
            complex(-0.49045898782913216, 0.503540043472969),
            complex(-1.5503995195448192, 1.4802696878897024),
            complex(1.8349919860385513, -0.6277553566871514),
            complex(1.4677459431657707, -0.28930671495252813),
            complex(-0.7254157547120995, 0.7981283614635981),
            complex(0.3452852632771768, -0.2643841796334996),
            complex(-1.4737822801997043, 1.1677654031709275),
            complex(-1.4198082512083805, -0.5271319461353843),
            complex(-2.01008763407216, 0.5557562139904978),
            complex(0.8979567558020968, 0.7533746148326915),
            complex(0.3803199454634167, -1.5997085756211813),
            complex(1.534409973458573, 1.601931298677989),
            complex(-0.02554495883306937, 0.1195904416941107),
            complex(1.2410592448091113, -1.5718581061863979),
            complex(-1.2000933156732572, 0.6999871399527559),
            complex(-0.5841319775257666, -1.0355650975335597),
            complex(0.7481814129562856, -0.49918952292056784),
            complex(-1.6321689153155512, 0.914589107016013),
            complex(-1.8189443356406587, 0.9412960722860186),
            complex(1.5314248132642987, -0.49955772185128067),
            complex(1.1252672597220157, 0.48811372912676665),
            complex(-0.8599552610491717, -1.7588618835108973),
            complex(-1.2974918398874686, 1.797354100781488),
            complex(0.8955926680615787, -0.9829072976619648),
            complex(-1.0332754389772578, -0.25584075397256806),
            complex(-0.1381274692350594, 0.721070163267276),
            complex(-1.1211001505412905, 1.4238118299771698),
            complex(-1.645711206031928, 0.3505597878390415),
            complex(-1.772336601753454, 0.18875445318895703),
            complex(-1.141050879835292, -0.5933948446762123),
            complex(1.5530582982342984, 0.8790585110831283),
            complex(-0.6159463207270269, 0.8431119714497369),
            complex(0.44681545287015295, -1.1502529649183986),
            complex(-0.8037369274290056, -0.7507509511576326),
            complex(0.8604249890162214, -0.9591556374156615),
            complex(0.22312150210690002, 0.1955991616646355),
            complex(1.544436444419624, -0.6593057406416784),
            complex(1.7812584136240874, -0.9905223310473721),
            complex(-0.4803877948383143, 1.0327468673522207),
            complex(0.43806343939991077, -0.843213470555268),
            complex(0.8050267618066825, 1.2871116591266978),
            complex(0.8667713264901313, -1.1273199994115377),
            complex(-1.7836681713782552, -0.21749723621725597),
            complex(-0.3402676358683345, 0.5825582948473786),
            complex(-1.541620275948796, -1.302128653419533),
            complex(0.8190297076100799, 0.09191446905963861),
            complex(0.5809368618035785, -0.7194104950785819),
            complex(-0.6235601124627498, -2.095070276698008),
        ]),
        result=np.array([
            complex(-0.02780298907988301, 0.06097948493231578),
            complex(-505.38643727215646, -973.9858917235066),
            complex(0.24534101222125634, 0.126523661489429),
            complex(3.4333897059894256, -8.828843076915161),
            complex(0.08170723837892624, -0.04624441531176056),
            complex(-0.09058315606734824, -0.0538914262773388),
            complex(-0.040517630839772334, -0.022404560041675292),
            complex(0.014267074833775449, -0.029149965648715147),
            complex(0.26896444297180766, 1.3619338891888895),
            complex(-0.05174479380461433, 0.020859739407209803),
            complex(-0.02052500475467127, -0.05510906562350468),
            complex(3.7601739878073452, 3.2828672515208277),
            complex(-0.06759667571437931, 0.0013036736393906966),
            complex(-0.027085843078090418, 0.08514728234035253),
            complex(0.013301450808462477, 0.015570935678020124),
            complex(-0.1628484612515275, 0.5084852472132736),
            complex(0.09144412359772079, 0.10614894115762075),
            complex(-0.0072440231039740365, 0.026596175911076245),
            complex(-140.03817953700644, 158.16777935651567),
            complex(4.173646255506006, -46.748673390820116),
            complex(0.02909506189920788, 0.08354305774248187),
            complex(0.6465929146265819, -0.5139171232705703),
            complex(0.0017072358436311316, 0.031243368772331793),
            complex(0.026761379027619377, -0.023821535669088564),
            complex(0.011885261300334293, 0.011437178134810704),
            complex(-2.037527361580098, 1.330458633539319),
            complex(-0.19371154933829324, 0.025962112001692197),
            complex(0.2458278269879398, 0.650434855234249),
            complex(0.3532805866012906, 0.09089160744022756),
            complex(0.35625187119194357, 0.2840678364760368),
            complex(0.02613646564900861, 0.039202266522934875),
            complex(-0.005691830851428308, -0.09743722517410515),
            complex(-0.03396013985070165, -2.156606220822054),
            complex(0.009501370559059023, 0.024913514761688273),
            complex(0.0070518178007279025, 0.02015375291553477),
            complex(-43.329442380912205, -23.65431702244191),
            complex(-3.2116712289881706, 7.370904551351257),
            complex(-0.03370644016892402, -0.03630629577628879),
            complex(-0.01901138186327002, 0.026610796953883596),
            complex(-1.5515999660468314, 0.11804449315710519),
            complex(0.05850516605832106, -0.023296182285175597),
            complex(0.04240362676883597, 0.2289319398350915),
            complex(-0.013157977160222906, 0.04145845928059304),
            complex(0.02359116649113966, 0.01232727309646365),
            complex(0.021630758792670533, 0.005776038828769675),
            complex(0.03433850743095363, -0.038899629790520224),
            complex(-2.711212823866397, -22.2137635716966),
            complex(0.023075657014811034, 0.09922213066589858),
            complex(-0.4013028463714598, -0.163100838472457),
            complex(0.03335868146662372, -0.07306356357315072),
            complex(-1.487059229836002, -0.09008929225059761),
            complex(0.5596197074235577, 0.28149385912005476),
            complex(-36.27234977318477, 12.902894574636496),
            complex(47.03047040494903, 15.942838724528073),
            complex(-0.012285619256473958, 0.11223572933244674),
            complex(-0.3456298308834066, -0.5506729552469383),
            complex(-0.5388853249525809, -0.3284458503555432),
            complex(-0.9764520066277106, 0.3773565846347695),
            complex(0.021060568678345905, -0.006527630045562824),
            complex(0.08797412459887761, 0.15173002937386723),
            complex(-0.0023781412744257584, -0.02842055330698165),
            complex(3.3068799651696064, 1.0657003527411557),
            complex(-0.4530958820323075, -0.9568301626153166),
            complex(-0.046509261570900684, -0.020177126644474717),
        ]),
    ),
    Reference(
        alpha=0.804498,
        beta=1.0,
        z=np.array([
            complex(1.128564388108359, -1.4649689098298009),
            complex(0.7583124507246032, -1.7946041327017042),
            complex(1.3426204422848682, 0.03703931219021064),
            complex(-0.9569187195836513, -0.06433344851009565),
            complex(-1.7392195014175833, 0.15486822402855846),
            complex(-1.2733695240255611, -0.199628562483197),
            complex(-1.8163453181559115, 1.0221536023634397),
            complex(-1.047357788501286, -1.6466984493800838),
            complex(-0.6834212528516888, -2.0910047847664703),
            complex(-0.00701448336478, 0.5795014534838792),
            complex(-0.5290741181549196, 0.6627117323739581),
            complex(0.5549948358250945, -1.2833017469927173),
            complex(-1.7413520551575787, 0.4256387355224492),
            complex(-2.0646286361757387, -0.7771152608503097),
            complex(-1.3132911406538597, -0.6252940355788512),
            complex(1.2359230592178454, -1.7196206556678695),
            complex(0.23102132834275288, 1.5328290075892228),
            complex(0.6571893130913667, -1.2273482719015343),
            complex(0.33532148613350754, -1.7947701012798989),
            complex(-0.6272120752857309, -2.1080189702036543),
            complex(1.1945162416108557, -1.5273804813011156),
            complex(-0.8354127007639461, 0.3626622969251284),
            complex(0.6345447172591709, -1.4422286792440564),
            complex(-1.7675740147019068, -0.948450991041309),
            complex(-0.8685407385341477, -1.399602125509227),
            complex(0.4033497334872498, 0.08993116272189008),
            complex(1.1642392915171824, 0.17266816085488496),
            complex(0.1375554921296313, -0.6284022280266948),
            complex(-0.9234829093020775, 1.038895560218623),
            complex(0.4740285972236946, 0.5843068283531485),
            complex(-1.5763023459089225, 0.53953364112858),
            complex(0.045645301249088464, -2.1872191771466367),
            complex(-0.6760576369731512, 0.33887952791091525),
            complex(0.664995875314172, 0.781247981709143),
            complex(-1.8368356953642353, -0.3775978795707171),
            complex(-1.3436491639899295, 0.7493785481518822),
            complex(0.13854061412377902, 0.80274312904465),
            complex(0.036691715130343465, -1.595576752517866),
            complex(0.06133724811228069, 2.0539010827598365),
            complex(-0.5891005140421665, -1.030037856680634),
            complex(0.47932884217039634, 0.29243683313396707),
            complex(1.7238231608472412, 0.35081675459481176),
            complex(1.1603600945562609, -1.650672522950073),
            complex(-0.01257831666428466, -0.8451360438945136),
            complex(0.6482857912929504, 0.985529599107858),
            complex(0.5283332581902115, -1.928923176038438),
            complex(1.1644655262683006, -0.5897833599548347),
            complex(-0.013304849047771896, 0.5077395118612015),
            complex(0.14456239036302979, -1.1160833944642334),
            complex(0.11652989857748712, -0.4397481420041856),
            complex(-1.0620514282091045, -1.280095016789872),
            complex(0.2893628384091845, -1.6580020253297174),
            complex(-0.7981496508432541, 0.7341727890449432),
            complex(1.067427657231264, -1.653157066803426),
            complex(1.3601674063263538, 1.1908470742282886),
            complex(-0.20627487070399306, -0.30229442297638),
            complex(-0.17283313513153317, 1.1851372187623823),
            complex(0.35054710888325874, -0.7167505267069318),
            complex(-0.714027331954383, -1.1540071863205588),
            complex(-0.6631207976396478, 1.9103352543503496),
            complex(0.658413229856289, 0.8113730023963952),
            complex(0.43440263758043657, -1.0785539310960652),
            complex(-1.1660601095366718, 0.7691923352816853),
            complex(0.5167893793659984, 0.07265613644526704),
        ]),
        result=np.array([
            complex(-1.1938848998052949, -2.894345980910251),
            complex(-1.1048928797480537, -1.2865220706203924),
            complex(5.165445191580782, 0.26126648803331226),
            complex(0.3997836925953684, -0.021625132019099946),
            complex(0.22105305686596646, 0.022802348238915384),
            complex(0.3056892692510142, -0.047092558323307),
            complex(0.15113286894509287, 0.11551942028144326),
            complex(0.06478469139297693, -0.25777788289564185),
            complex(-0.08587435656096495, -0.27192710431757694),
            complex(0.7751514571904424, 0.5546578101259332),
            complex(0.4506053395676911, 0.32525100580107097),
            complex(0.023637104784474006, -1.560536154033739),
            complex(0.20970291727473905, 0.0606963834225823),
            complex(0.15338579068850872, -0.07619714891477906),
            complex(0.256768674645534, -0.13095797574386167),
            complex(-2.3624190928493074, -2.3872010844184977),
            complex(-0.1707208841762093, 0.9568387020763306),
            complex(0.10058064506109005, -1.7983045056312759),
            complex(-0.5017052169947206, -0.8518290046425677),
            complex(-0.10324546756090107, -0.27815789076021924),
            complex(-1.6057422137457098, -2.959687322350949),
            complex(0.415857418216046, 0.13553727885530356),
            complex(-0.3393628440602826, -1.605650893575289),
            complex(0.16245836760212196, -0.11512417033587774),
            complex(0.12400960678368941, -0.3149077654942958),
            complex(1.560961972586404, 0.16442291931089273),
            complex(3.94926318588435, 0.9232160189652939),
            complex(0.850875073464702, -0.7101576899782926),
            complex(0.23110736028976, 0.27766013203703555),
            complex(1.2640190590127185, 1.0484211743754446),
            complex(0.22438292479171262, 0.08847796282577922),
            complex(-0.4294466094416446, -0.37477229626289443),
            complex(0.4811711082669121, 0.15283962373153306),
            complex(1.151169729844865, 1.646552305566359),
            complex(0.1999672725588852, -0.049496596814734894),
            complex(0.2342534704180246, 0.1465814129406998),
            complex(0.6781469083221412, 0.8364695502493927),
            complex(-0.12976228500779238, -0.747962092594507),
            complex(-0.4067335248134344, 0.4769783263307821),
            complex(0.2804220936782523, -0.3969620793726936),
            complex(1.6002116982669499, 0.577699405285068),
            complex(7.643466587929043, 4.208341970156063),
            complex(-1.8899089079463467, -2.4436078521452744),
            complex(0.5583352286609543, -0.712006851174142),
            complex(0.6662697313834611, 1.7694191236016903),
            complex(-0.839988027614925, -0.8367301939401973),
            complex(2.7785931775712873, -2.7808865120128616),
            complex(0.8181218872955237, 0.4940426711115189),
            complex(0.32584988021414524, -0.9538941879382805),
            complex(0.9816986607127026, -0.5162225837295499),
            complex(0.15432464874584476, -0.25855811354294317),
            complex(-0.337602050118165, -0.9320037485588876),
            complex(0.34387509397707805, 0.25797076905353955),
            complex(-1.600762551194787, -2.190706575847162),
            complex(-0.3272055555660725, 4.62190724804732),
            complex(0.7590065879748815, -0.2424112711770301),
            complex(0.23711208380998494, 0.6592261974089232),
            complex(0.939897877133936, -1.0271765610335961),
            complex(0.2145008042390787, -0.36019991163256965),
            complex(-0.050484977030355455, 0.31792062487051465),
            complex(1.0763659548154025, 1.6645231434617263),
            complex(0.4169837775992258, -1.3628110970444507),
            complex(0.2590704242179087, 0.17939148732388),
            complex(1.7882378753836614, 0.15534952104961305),
        ]),
    ),
    Reference(
        alpha=0.988201,
        beta=1.0,
        z=np.array([
            -1.8474470856250311,
            -2.3985372495237667,
            -3.6008077486588084,
            -4.716459847792277,
            -1.1636783366520695,
            -4.4886884258422075,
            -4.598086088328206,
            -3.800496742825555,
            -4.212298617702749,
            -4.6192431132743055,
            -2.9072858358594527,
            -3.2942890210043188,
            -4.335180862721412,
            -2.116803374714642,
            -2.2515122163221806,
            -4.02117743569097,
            -3.4766110775728993,
            -1.6392612168577132,
            -2.079376426664416,
            -4.153256679080328,
            -2.8426557151896548,
            -2.984541208748278,
            -3.928523677746317,
            -3.836779446877862,
            -0.707887119542017,
            -3.0910877744055183,
            -3.2628304067850133,
            -1.49208663442829,
            -3.7851581344428404,
            -3.176853104406035,
            -4.514333459149146,
            -4.463347788601207,
            -2.705052341806518,
            -3.0683506780893417,
            -2.7588951024547144,
            -4.2556240112246035,
            -3.9393656715744076,
            -2.147850618466798,
            -3.430539598007514,
            -2.891443065434584,
            -2.1654647821460995,
            -1.146768527926589,
            -1.608082889732268,
            -3.6547315014345614,
            -4.81140353226769,
            -3.42813414341914,
            -2.495767762138953,
            -3.586790555975951,
            -3.283728992168157,
            -3.8795173399182463,
            -2.2933014664152807,
            -3.8683795364720517,
            -4.332568860195912,
            -3.2212430955223867,
            -3.3683015680982082,
            -2.5561392994289647,
            -3.2313441776891945,
            -4.066610650358403,
            -4.333671167741894,
            -3.4592721470859114,
            -3.8071255342574553,
            -4.721742697170763,
            -2.7678255335168407,
            -3.9650617528333765,
        ]),
        result=np.array([
            0.16074786822191908,
            0.09480543517565226,
            0.03158837492106241,
            0.012688892830432712,
            0.3136374411391023,
            0.015112763781518443,
            0.01388421322928068,
            0.026580373926185202,
            0.018843229396274207,
            0.013660782180118661,
            0.05891648552939276,
            0.04143428908484097,
            0.017064046027460693,
            0.1240080010595442,
            0.10902285402715606,
            0.022058199219576833,
            0.03522815013853452,
            0.19676377747833262,
            0.12854127486568917,
            0.019774711903809538,
            0.06253988973404143,
            0.054877113793064855,
            0.02384102620976202,
            0.025769520628577763,
            0.4926201416162225,
            0.04978619138825838,
            0.0426212296114601,
            0.22716441544352978,
            0.026931773267194467,
            0.04605765255254207,
            0.0148133687060872,
            0.0154157691318017,
            0.07106846106827444,
            0.05082836203055611,
            0.06759274188199133,
            0.01819216467951581,
            0.02362414070479979,
            0.12037377010207786,
            0.036693947371836864,
            0.05978346629078897,
            0.11836141426351529,
            0.31891159747272635,
            0.20283447148084732,
            0.030139443336837123,
            0.011820767179461896,
            0.03677229759003765,
            0.08648169780810631,
            0.03197756862498147,
            0.041828662709117996,
            0.024849997428042902,
            0.10476818746092705,
            0.02508600529450466,
            0.017099752436622095,
            0.04424737630690452,
            0.03878138028208125,
            0.08170313000330082,
            0.04384630487167113,
            0.02124004983601248,
            0.017084673004156517,
            0.035772109754867845,
            0.02643011064964003,
            0.012638554878280795,
            0.06703391224611394,
            0.023119022371003203,
        ]),
    ),
]
