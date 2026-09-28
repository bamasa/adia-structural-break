# --- competition interface ---------------------------------------------------

#: The dilated-convolution channel's fitted weights, embedded so the submission
#: has no torch dependency and no external file: 721 parameters, trained once
#: locally with a fixed seed, forward pass reimplemented in numpy below and
#: asserted against torch at export time.
CNN_WEIGHTS = json.loads(r"""{"weights": [[[[-0.07266596704721451, 0.36535537242889404, -0.38401609659194946, -0.28864219784736633, -0.1740281879901886]], [[0.04633694887161255, -0.2653433382511139, 0.3749937117099762, -0.08925638347864151, 0.11243904381990433]], [[-0.18263278901576996, 0.327459454536438, -0.4635157287120819, -0.31172114610671997, -0.09763425588607788]], [[-0.1484927535057068, 0.319388210773468, 0.3934409022331238, -0.19953441619873047, -0.3052457869052887]], [[0.1963605135679245, 0.4150046706199646, 0.16825689375400543, 0.3154437243938446, 0.016427600756287575]], [[0.008882584981620312, 0.39959582686424255, -0.5141045451164246, -0.24813681840896606, -0.13653503358364105]], [[-0.03034358099102974, 0.2227049022912979, 0.02118147537112236, -0.05396464839577675, -0.09483098983764648]], [[-0.3941996395587921, -0.019504593685269356, 0.3313857614994049, 0.2779979407787323, 0.1430901139974594]]], [[[-0.059138134121894836, 0.04617363214492798, 0.15371888875961304, 0.17044954001903534, 0.0798509269952774], [-0.04190114885568619, -0.016669372096657753, -0.2779388129711151, -0.11069845408201218, -0.23617416620254517], [-0.10453129559755325, -0.039935559034347534, 0.10067954659461975, 0.07706207036972046, -0.08804690837860107], [-0.1130729615688324, -0.11141818016767502, -0.22384576499462128, -0.12037637084722519, -0.05234265327453613], [0.08196742832660675, 0.0859467014670372, -0.13212136924266815, -0.17143306136131287, 0.02716292254626751], [0.12277842313051224, 0.13620881736278534, 0.16621318459510803, -0.02597498893737793, -0.1505914032459259], [0.06804097443819046, -0.03308410197496414, -0.06715670228004456, 0.19195275008678436, 0.1749388873577118], [-0.09446234256029129, 0.11104535311460495, 0.0010715776588767767, 0.06990113109350204, -0.010398758575320244]], [[0.07344873249530792, -0.008373993448913097, 0.1024591401219368, 0.12097416818141937, 0.1271982342004776], [-0.045978568494319916, -0.09013072401285172, -0.357136994600296, -0.1608702391386032, -0.13989733159542084], [-0.09329688549041748, -0.029072577133774757, 0.16961684823036194, 0.14916692674160004, -0.0660950243473053], [-0.2386324554681778, -0.3692983388900757, -0.27624648809432983, -0.2268332988023758, -0.2869633138179779], [-0.028568744659423828, -0.021838944405317307, -0.06358036398887634, -0.010861807502806187, -0.02288881689310074], [0.10633326321840286, 0.1452413946390152, -0.1189064234495163, -0.09047595411539078, -0.05856044217944145], [0.08933336287736893, 0.13315947353839874, 0.16477148234844208, 0.08561810106039047, 0.02134598046541214], [0.16016885638237, 0.010045374743640423, 0.015190277248620987, 0.05346348136663437, 0.02399212121963501]], [[0.07522007077932358, -0.06560633331537247, 0.006296883337199688, 0.07098760455846786, -0.06372906267642975], [-0.3095565140247345, 0.04410812258720398, -0.2428840547800064, 0.3187541663646698, 0.11791616678237915], [0.15452903509140015, 0.14923174679279327, 0.15936680138111115, 0.1255631446838379, 0.16919894516468048], [-0.12204264849424362, -0.15758198499679565, 0.05179924890398979, 0.08420871943235397, 0.07260526716709137], [0.004545318894088268, -0.12207717448472977, 0.1755431592464447, -0.01996743679046631, 0.09010405838489532], [0.26369088888168335, 0.11848132312297821, -0.13378682732582092, 0.10232344269752502, -0.008707409724593163], [0.058634497225284576, -0.015890374779701233, 0.037288784980773926, 0.02100549079477787, -0.14709416031837463], [-0.19582800567150116, 0.04520189017057419, -0.03734823688864708, 0.10788774490356445, 0.17218708992004395]], [[-0.1469256728887558, -0.03460075333714485, 0.11763176321983337, -0.10897347331047058, 0.1255406141281128], [0.017811216413974762, -0.1832926720380783, 0.12980933487415314, -0.05878594145178795, -0.10419053584337234], [0.11181643605232239, -0.14218145608901978, -0.006225927267223597, -0.08371762186288834, 0.0945466160774231], [0.007023524958640337, 0.15039263665676117, 0.10999167710542679, -0.004907877650111914, -0.08342748135328293], [-0.16691669821739197, 0.1870676577091217, 0.1041833758354187, 0.046802956610918045, -0.14154376089572906], [-0.015018744394183159, -0.1329239308834076, -0.14711043238639832, -0.15105514228343964, 0.05826893076300621], [-0.11405204981565475, -0.013636019080877304, -0.01819911226630211, 0.06773041933774948, -0.15721693634986877], [-0.14044244587421417, -0.10739696770906448, 0.09977449476718903, -0.1199827641248703, 0.1388711780309677]], [[0.05834551900625229, 0.1381978988647461, 0.062259841710329056, -0.008133233524858952, 0.07512498646974564], [-0.15940724313259125, -0.0661868155002594, -0.3134287893772125, -0.5093613862991333, -0.2922689914703369], [0.13668856024742126, -0.03620199114084244, 0.0053617628291249275, 0.21174481511116028, 0.08177337050437927], [-0.16928139328956604, -0.16492584347724915, 0.042284730821847916, -0.14655481278896332, -0.13918302953243256], [-0.23031826317310333, -0.13709448277950287, -0.11603778600692749, -0.1464557647705078, -0.09703379124403], [0.09843853861093521, 0.11915268003940582, -0.07204365730285645, -0.010665013454854488, -0.09813082963228226], [-0.04260752350091934, 0.17433878779411316, 0.17139527201652527, 0.1617736518383026, 0.0459241047501564], [0.13240410387516022, 0.20853592455387115, 0.033593952655792236, 0.21354402601718903, -0.049942538142204285]], [[0.019658450037240982, 0.003093617269769311, -0.007402326911687851, 0.07788045704364777, 0.15720833837985992], [-0.24156668782234192, -0.28418561816215515, -0.3480963408946991, -0.34211021661758423, -0.2311682403087616], [-0.0488220676779747, -0.10727489739656448, -0.058174069970846176, 0.1508089154958725, 0.08641470968723297], [-0.06066516414284706, -0.11866629868745804, -0.08814534544944763, -0.0238803718239069, -0.047468531876802444], [0.13095104694366455, -0.06154302507638931, -0.184219092130661, -0.14862747490406036, -0.22847095131874084], [-0.007237298414111137, -0.13106945157051086, -0.10580844432115555, 0.12543489038944244, -0.1539032757282257], [0.05114755034446716, 0.11914321780204773, 0.14097534120082855, -0.018614796921610832, -0.027333194389939308], [0.2002330869436264, 0.1034429669380188, 0.21192242205142975, 0.0347059890627861, 0.06734433025121689]], [[0.005103309638798237, 0.026759859174489975, -0.05924120172858238, 0.1270555853843689, 0.04089225456118584], [-0.1919938623905182, -0.07990031689405441, -0.33878856897354126, -0.16975075006484985, -0.24093687534332275], [0.10940025001764297, -0.0325421579182148, -0.07665270566940308, 0.09953749924898148, -0.0949660912156105], [-0.1287786066532135, -0.19313979148864746, -0.2027260810136795, -0.11111266911029816, -0.12527219951152802], [-0.03707502782344818, -0.12918078899383545, -0.20530784130096436, -0.048512108623981476, 0.03946112096309662], [-0.07407931983470917, 0.043226562440395355, 0.07434353977441788, -0.0547136627137661, -0.07380810379981995], [0.13823635876178741, 0.07611194252967834, 0.1257246732711792, 0.11143500357866287, -0.05150715261697769], [0.08468808978796005, 0.18303436040878296, 0.22740834951400757, 0.14765281975269318, 0.08809313923120499]], [[0.19747118651866913, -0.024803102016448975, -0.021751029416918755, -0.34706833958625793, 0.25789499282836914], [-0.009912432171404362, -0.4104403853416443, 0.22023265063762665, 0.11552625149488449, 0.11448077112436295], [0.20899517834186554, 0.17008304595947266, 0.12860535085201263, -0.29257452487945557, 0.20288920402526855], [0.15956231951713562, -0.13583636283874512, 0.1851138174533844, -0.16067203879356384, 0.3064350187778473], [-0.13668222725391388, 0.09105762094259262, -0.15159691870212555, 0.34594377875328064, -0.03269543498754501], [0.265011191368103, 0.031092967838048935, 0.1576252281665802, -0.3240121603012085, 0.20065492391586304], [-0.04558062553405762, 0.10389783978462219, -0.12190999835729599, -0.18767249584197998, -0.0849926695227623], [0.14768218994140625, 0.07740621268749237, 0.13255666196346283, 0.20392796397209167, -0.14635664224624634]]], [[[0.15786750614643097, 0.1666419506072998, 0.1527860462665558, 0.07780880481004715, 0.1589934527873993], [0.0474998839199543, 0.13993918895721436, 0.22725574672222137, 0.0920000746846199, 0.22787976264953613], [0.15015222132205963, 0.1964368224143982, -0.01945049874484539, 0.09731553494930267, 0.025496596470475197], [-0.007236317731440067, -0.038450006395578384, -0.033755846321582794, -0.08465082943439484, 0.1431020051240921], [0.20748209953308105, 0.04205714538693428, -0.035411935299634933, 0.17469021677970886, 0.20409195125102997], [0.06321407854557037, -0.05423946678638458, 0.19933681190013885, 0.1498749703168869, -0.13791941106319427], [0.20010848343372345, 0.1691150814294815, -0.02845182456076145, 0.14229176938533783, -0.052663058042526245], [-0.012722316198050976, 0.10243749618530273, 0.001169963856227696, -0.1227884441614151, -0.07331733405590057]], [[-0.048742473125457764, -0.027551086619496346, 0.01849263347685337, 0.1457531750202179, 0.0025172766763716936], [-0.06126939132809639, 0.06010901555418968, 0.14580538868904114, 0.1461813747882843, -0.06655426323413849], [0.006396281532943249, -0.07718757539987564, -0.04858068376779556, 0.06575117260217667, -0.0963163748383522], [0.12261679023504257, 0.18965034186840057, 0.04085325822234154, -0.06408834457397461, -0.009934160858392715], [0.0672948881983757, -0.05865666642785072, 0.14132842421531677, -0.07288219779729843, -0.07111137360334396], [-0.11937816441059113, 0.02215956151485443, 0.15601547062397003, -0.09123306721448898, 0.19101405143737793], [-0.037016287446022034, 0.087050661444664, 0.05021705478429794, -0.10471190512180328, 0.1821148544549942], [0.11229751259088516, 0.15994995832443237, 0.0723111629486084, -0.1036158949136734, 0.11960742622613907]], [[-0.04338812455534935, 0.019333574920892715, -0.17513011395931244, -0.09161722660064697, -0.2032078504562378], [-0.11074882745742798, -0.020161768421530724, -0.17177219688892365, -0.001693509635515511, 0.0338299423456192], [-0.04082900658249855, -0.16209733486175537, -0.17788773775100708, -0.10433219373226166, 0.00146418996155262], [0.11673115193843842, 0.054610561579465866, -0.056259673088788986, 0.018007157370448112, -0.19105201959609985], [-0.1781725287437439, 0.02178489789366722, -0.1310410499572754, 0.027644028887152672, -0.15527212619781494], [-0.12982052564620972, -0.17946144938468933, 0.028999846428632736, -0.06735968589782715, 0.06781486421823502], [0.044182732701301575, 0.031522706151008606, -0.04753433167934418, -0.10175201296806335, -0.08642755448818207], [-0.0280794445425272, 0.029198849573731422, -0.15364298224449158, 0.07744250446557999, -0.06352938711643219]], [[-0.10916629433631897, -0.2203637957572937, 0.0828181579709053, 0.07932247966527939, -0.13136771321296692], [0.11602295935153961, -0.031054237857460976, 0.019948581233620644, 0.0592438206076622, -0.01968088559806347], [0.2507339417934418, 0.2316712588071823, -0.04885419085621834, 0.013864870183169842, -0.10004783421754837], [0.03996431455016136, -0.0879625529050827, -0.09826092422008514, -0.001657521235756576, 0.007928467355668545], [-0.08729220926761627, -0.17018118500709534, 0.03750700503587723, -0.07268108427524567, 0.02858889475464821], [-0.12550956010818481, -0.05143796652555466, -0.24679657816886902, -0.19236774742603302, -0.09089773148298264], [0.11202679574489594, -0.22111260890960693, -0.14354851841926575, -0.12777067720890045, 0.09025201946496964], [0.1848336011171341, -0.015352067537605762, 0.29867708683013916, 0.10450126230716705, -0.06732380390167236]], [[-0.09760520607233047, -0.09658834338188171, -0.10860779881477356, 0.05538708344101906, -0.08042702078819275], [0.14025425910949707, 5.312991561368108e-05, -0.03448440879583359, 0.07998818159103394, 0.23331481218338013], [-0.09606583416461945, 0.07872532308101654, -0.12803083658218384, 0.1909063309431076, -0.228504940867424], [-0.09277703613042831, 0.25575119256973267, -0.07136852294206619, 0.09473881125450134, 0.01932990737259388], [-0.015363908372819424, -0.04801918938755989, 0.1578189730644226, 0.06326418370008469, -0.05485735461115837], [-0.07208260893821716, 0.0009414826636202633, 0.13475026190280914, -0.08303821086883545, 0.16733644902706146], [-0.012352907098829746, -0.03954903036355972, 0.16131094098091125, 0.1244293749332428, 0.1587761789560318], [0.2382027506828308, -0.19462615251541138, 0.03407410532236099, -0.04263107478618622, 0.13053303956985474]], [[-0.25476059317588806, -0.35302430391311646, 0.0036195802967995405, 0.10342748463153839, -0.18155576288700104], [-0.1625356823205948, -0.35669752955436707, -0.03922716900706291, -0.09795164316892624, -0.05477699264883995], [0.24562585353851318, 0.24808020889759064, 0.2032395452260971, 0.057826586067676544, 0.1183033362030983], [-0.1295841783285141, -0.03982139006257057, 0.09571945667266846, 0.04573991522192955, 0.024474579840898514], [-0.2548828721046448, 0.08900532126426697, 0.06594540923833847, 0.037556152790784836, -0.25807690620422363], [-0.1371937394142151, -0.32285577058792114, -0.003265828127041459, -0.14432191848754883, -0.13819287717342377], [0.058110564947128296, -0.3276696801185608, -0.2641007602214813, -0.12498490512371063, -0.09363022446632385], [-0.11474420875310898, 0.3894168734550476, 0.25988060235977173, 0.06178545951843262, 0.17068222165107727]], [[0.15906712412834167, 0.21900688111782074, 0.20776914060115814, 0.038762662559747696, -0.05252693593502045], [0.11531481146812439, -0.05858748406171799, -0.2711677849292755, -0.22056205570697784, -0.33752408623695374], [0.009362898766994476, 0.060424912720918655, 0.049544643610715866, 0.12580782175064087, 0.1327829211950302], [0.08887876570224762, 0.08977971971035004, 0.571628212928772, 0.197941854596138, 0.07658220082521439], [-0.13492731750011444, -0.27900373935699463, -0.22081229090690613, -0.17847569286823273, -0.45357683300971985], [0.0014040963724255562, -0.07174360752105713, -0.10240492224693298, -0.327627956867218, -0.1360824853181839], [-0.10986284166574478, 0.0936395525932312, -0.20485025644302368, -0.1773754507303238, -0.12123966962099075], [0.13616904616355896, 0.35719072818756104, 0.5374603271484375, 0.2669313848018646, 0.43744808435440063]], [[-0.16165724396705627, 0.14369376003742218, -0.14443868398666382, 0.003228071378543973, 0.0003351292107254267], [-0.15679532289505005, 0.11264646053314209, -0.221826434135437, -0.008656497113406658, 0.1885976493358612], [-0.021417956799268723, 0.04708721116185188, -0.16957010328769684, 0.1626754254102707, -0.4001619517803192], [-0.04151744395494461, 0.08482171595096588, -0.04875239357352257, 0.17366383969783783, 0.20395909249782562], [-0.09773099422454834, -0.02719491720199585, 0.11750613898038864, 0.020400797948241234, 0.18236082792282104], [-0.23148775100708008, -0.04603837803006172, -0.02426370419561863, 0.030963433906435966, 0.15737831592559814], [-0.1915704905986786, -0.20932775735855103, 0.08988994359970093, 0.08159878849983215, -0.028392065316438675], [-0.23984111845493317, -0.006517020985484123, 0.21135753393173218, -0.26095348596572876, -0.24328725039958954]]]], "biases": [[0.14118149876594543, -0.5116382837295532, 0.14577384293079376, -0.5408392548561096, -0.35823559761047363, -0.19792646169662476, 0.5832894444465637, 0.4822920858860016], [0.018467577174305916, 0.17108038067817688, -0.04695342108607292, 0.08008492738008499, 0.1019597053527832, 0.08697020262479782, -0.0006312492769211531, -0.07398853451013565], [0.21935778856277466, 0.025616778060793877, -0.00841254647821188, -0.04282330721616745, 0.03571523725986481, -0.02544151060283184, -0.19449251890182495, 0.08638565242290497]], "head_w": [-0.2109154462814331, 0.006330783013254404, -0.012051859870553017, 0.21376824378967285, -0.16998203098773956, 0.2684285342693329, 0.19878853857517242, -0.16279537975788116, -0.2883453369140625, -0.21580877900123596, 0.20575323700904846, -0.1299976408481598, -0.029396722093224525, 0.03226596117019653, 0.30248162150382996, 0.20041824877262115], "head_b": 0.10040033608675003}""")

CNN_WINDOW = 128
CNN_DILATIONS = (1, 4, 16)
CNN_KERNEL = 5


class InlineCnn:
    def __init__(self, data: dict) -> None:
        self.weights = [np.asarray(w) for w in data["weights"]]
        self.biases = [np.asarray(b) for b in data["biases"]]
        self.head_w = np.asarray(data["head_w"])
        self.head_b = float(data["head_b"])

    def forward(self, window: np.ndarray) -> float:
        h = window.reshape(1, -1).astype("float64")
        for weight, bias, dilation in zip(self.weights, self.biases, CNN_DILATIONS):
            pad = (CNN_KERNEL - 1) * dilation // 2
            padded = np.pad(h, ((0, 0), (pad, pad)))
            out = np.empty((weight.shape[0], h.shape[1]))
            for j in range(h.shape[1]):
                taps = padded[:, j : j + (CNN_KERNEL - 1) * dilation + 1 : dilation]
                out[:, j] = np.tensordot(weight, taps, axes=([1, 2], [0, 1])) + bias
            h = np.maximum(out, 0.0)
        pooled = np.concatenate([h.max(axis=1), h.mean(axis=1)])
        z = float(pooled @ self.head_w + self.head_b)
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


VIEWS = ("raw", "whitened", "absolute")


class Monitor:
    """One series watched one observation at a time: all fifty channels."""

    def __init__(self, history: np.ndarray) -> None:
        self.norm = Normalisation.fit(history)
        self.detectors = {view: (Cusum(), PageHinkley(), VarianceRatio()) for view in VIEWS}
        self.multiscale = MultiScale()
        self.retro = Retro2.__new__(Retro2)
        # Retro2 refits the same normalisation; share it instead.
        self.retro.norm = self.norm
        self.retro.values = []
        self.retro.scores = dict.fromkeys(RETRO2_CHANNELS, 0.5)
        self.retro._argmax_trail = []
        self.retro._next_scan = 1
        self._previous_z = self.norm.standardise(float(history[-1]), -1) if len(history) else 0.0
        self._step = 0
        # The convolutional channel: a trailing window of standardised values
        # that may span the boundary -- the platform has the history in hand
        # too, and "the boundary looked ordinary" is itself a learnable shape.
        self.cnn = InlineCnn(CNN_WEIGHTS)
        tail = [
            self.norm.clip(self.norm.standardise(float(v), -len(history) + i))
            for i, v in enumerate(np.asarray(history, dtype="float64")[-CNN_WINDOW:])
        ]
        self._window = tail
        self._cnn_score = 0.5
        self._cnn_next = 1

    def update(self, x: float) -> list[float]:
        z = self.norm.clip(self.norm.standardise(float(x), self._step))
        w = whiten(z, self._previous_z, self.norm.rho)
        views = (z / self.norm.inflation, w, abs(w) - 0.7979)
        self._previous_z = z
        self._step += 1
        stream = [
            detector.update(value)
            for value, group in zip(views, (self.detectors[v] for v in VIEWS))
            for detector in group
        ]
        scales = self.multiscale.update(z)
        retro = self.retro.update(float(x))

        # The convolutional channel runs on the retro cadence: geometric early,
        # then sparse -- a forward pass per step would be quadratic in cost for
        # a score that changes little between adjacent late steps.
        self._window.append(z)
        if len(self._window) > CNN_WINDOW:
            self._window.pop(0)
        if self._step >= self._cnn_next:
            self._cnn_next = max(self._cnn_next + 1, int(self._cnn_next * 1.12))
            w = np.asarray(self._window, dtype="float64")
            if len(w) < CNN_WINDOW:
                w = np.concatenate([np.zeros(CNN_WINDOW - len(w)), w])
            self._cnn_score = self.cnn.forward(w)

        # The reverting counterparts, in the same detector order as `stream`:
        # the current statistic rather than the running peak. The peak cannot
        # recant a false alarm; the current value drains once the stream
        # behaves again, and the trees hold both.
        nows = [d.now for v in VIEWS for d in self.detectors[v]]

        return (
            stream
            + scales
            + [retro[c] for c in RETRO2_CHANNELS]
            + [self._cnn_score]
            + nows
        )


class DualMonitor:
    """Two full pipelines over two compressions of the same stream.

    The second sees asinh(x): log-like in the tails, linear near zero, defined
    on negatives. It is not a derived channel -- its normalisation, trend,
    thresholds and detectors are all fitted on the compressed series, so where
    heavy tails fool the raw view the two disagree, and the disagreement is
    what the combiner feeds on. Adopted on grouped 5-fold CV: 0.5746 against
    0.5719 for the raw-only fifty, ahead on three folds with no meaningful
    loss on the other two.
    """

    def __init__(self, history: np.ndarray) -> None:
        history = np.asarray(history, dtype="float64")
        self.raw = Monitor(history)
        self.compressed = Monitor(np.arcsinh(history))

    def update(self, x: float) -> list[float]:
        return self.raw.update(float(x)) + self.compressed.update(float(np.arcsinh(x)))


FORECAST_K = 10       # trailing-mean window
FORECAST_LAGS = 8
FORECAST_CHANNELS = 4


def _deviation_features(d: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Lag rows for the forecaster: eight trailing deviations plus the mean
    and spread of the last five. Purely backward-looking."""
    lags = np.stack([d[idx - j] for j in range(1, FORECAST_LAGS + 1)], axis=1)
    roll5 = np.stack([d[idx - j] for j in range(1, 6)], axis=1)
    return np.hstack([lags, roll5.mean(1, keepdims=True), roll5.std(1, keepdims=True)])


class Forecaster:
    """Is the next value still predictable the way the history was?

    A small LightGBM regressor predicts each next deviation-from-trailing-mean
    of the standardised stream: pretrained on every training history, then
    finetuned here on this series' own break-free history, where its residual
    scale sigma is measured. The reported channels are the standardised
    prediction error smoothed fast and slow, its running peak, and the raw
    step error. The EWMAs rise while the series stops being forecastable and
    drain once it is again -- an organic, reversible alarm.
    """

    def __init__(self, base, norm: Normalisation, history: np.ndarray) -> None:
        import lightgbm as lgb

        n = len(history)
        z_hist = [norm.clip(norm.standardise(float(v), i - n)) for i, v in enumerate(history)]
        self._z = list(z_hist)          # full standardised past, history then online
        self._steps_done = 0
        d = self._deviations(np.asarray(self._z))
        idx = np.arange(FORECAST_LAGS + FORECAST_K, n)
        self.model, self.sigma = base, 1.0
        if len(idx) > 40:
            feats = _deviation_features(d, idx).astype("float32")
            targets = d[idx].astype("float32")
            self.model = lgb.train(
                dict(objective="l2", learning_rate=0.03, num_leaves=7,
                     min_data_in_leaf=20, verbose=-1, deterministic=True,
                     force_row_wise=True, num_threads=1, seed=0),
                lgb.Dataset(feats, targets), num_boost_round=30, init_model=base)
            resid = targets - self.model.predict(feats, num_threads=1)
            self.sigma = float(np.std(resid)) + 1e-6
        self.fast, self.slow, self.peak = 1.0, 1.0, 0.0

    @staticmethod
    def _deviations(z: np.ndarray) -> np.ndarray:
        csum = np.concatenate([[0.0], np.cumsum(z)])
        out = np.full(len(z), np.nan)
        for i in range(FORECAST_K, len(z)):
            out[i] = z[i] - (csum[i] - csum[i - FORECAST_K]) / FORECAST_K
        return out

    def update(self, z: float) -> list[float]:
        self._z.append(float(z))
        arr = np.asarray(self._z)
        d = self._deviations(arr[-(FORECAST_K + FORECAST_LAGS + 2):])
        i = len(d) - 1
        error = 0.0
        row = _deviation_features(d, np.asarray([i]))
        if np.isfinite(row).all() and np.isfinite(d[i]):
            pred = float(self.model.predict(row, num_threads=1)[0])
            error = abs(float(d[i]) - pred) / self.sigma
        e = min(error, 8.0)
        self.fast += 0.10 * (e - self.fast)
        self.slow += 0.02 * (e - self.slow)
        self.peak = max(self.peak, self.fast)
        return [self.fast, self.slow, self.peak, e]


BATTERY_QS = (0.05, 0.25, 0.50, 0.75, 0.95)


def _acf(v: np.ndarray, lag: int) -> float:
    if len(v) <= lag + 2:
        return 0.0
    a, b = v[:-lag], v[lag:]
    sa, sb = a.std(), b.std()
    if sa < 1e-9 or sb < 1e-9:
        return 0.0
    return float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb))


def _hist_summary(z: np.ndarray):
    return (np.sort(z), np.quantile(z, BATTERY_QS), float(z.mean()), float(z.std()),
            float(np.median(np.abs(z - np.median(z)))),
            _acf(z, 1), _acf(z, 2), _acf(z, 5), float(np.abs(z).mean()))


def _two_sample(summary, prefix: np.ndarray) -> list:
    """Drift-free two-sample statistics: the history against the prefix so
    far. No length multipliers anywhere -- under the null every statistic is
    centred regardless of how long the prefix has grown, so length is never
    read as evidence."""
    from scipy.special import erfinv

    hist_sorted, hq, hmean, hstd, hmad, hacf1, hacf2, hacf5, habs = summary
    p = prefix
    n = len(p)
    out = []
    out.extend((np.quantile(p, BATTERY_QS) - hq).tolist())
    out.append(float(p.mean() - hmean))
    out.append(float(np.log((p.std() + 1e-9) / (hstd + 1e-9))))
    c = p - p.mean()
    s2 = float((c ** 2).mean()) + 1e-12
    out.append(float((c ** 3).mean()) / s2 ** 1.5)
    out.append(float((c ** 4).mean()) / s2 ** 2 - 3.0)
    pos = np.searchsorted(hist_sorted, np.sort(p), side="right") / len(hist_sorted)
    out.append(float(np.max(np.abs(pos - (np.arange(1, n + 1) / n)))))
    pmad = float(np.median(np.abs(p - np.median(p)))) + 1e-9
    out.append(float(np.log(pmad / (hmad + 1e-9))))
    out.append(_acf(p, 1) - hacf1)
    out.append(_acf(p, 2) - hacf2)
    out.append(_acf(p, 5) - hacf5)
    out.append(float(np.abs(p).mean() - habs))
    u = (np.searchsorted(hist_sorted, p, side="left")
         + np.searchsorted(hist_sorted, p, side="right")) / 2.0
    u = np.clip((u + 0.5) / (len(hist_sorted) + 1.0), 1e-6, 1 - 1e-6)
    g = np.sqrt(2) * erfinv(2 * u - 1)
    out.append(float(g.mean()))
    out.append(float(np.log(g.std() + 1e-9)))
    tail = p[-max(n // 4, 5):]
    out.append(float(tail.mean() - hmean))
    out.append(float(np.log((tail.std() + 1e-9) / (hstd + 1e-9))))
    half = n // 2
    if half >= 3:
        out.append(float(p[half:].mean() - p[:half].mean()))
        out.append(float(np.log((p[half:].std() + 1e-9) / (p[:half].std() + 1e-9))))
    else:
        out.extend([0.0, 0.0])
    return out


def _acf1(v: np.ndarray) -> float:
    if len(v) <= 3:
        return 0.0
    a, b = v[:-1], v[1:]
    sa, sb = a.std(), b.std()
    if sa < 1e-9 or sb < 1e-9:
        return 0.0
    return float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb))


def _slope_t(v: np.ndarray) -> float:
    n = len(v)
    if n < 8:
        return 0.0
    t_ax = np.arange(n) - (n - 1) / 2.0
    denom = float((t_ax ** 2).sum())
    beta = float((t_ax * (v - v.mean())).sum()) / denom
    resid = v - v.mean() - beta * t_ax
    se = np.sqrt(float((resid ** 2).sum()) / max(n - 2, 1) / denom) + 1e-12
    return float(np.clip(beta / se, -12, 12))


def _hist_summary2(z: np.ndarray) -> dict:
    d = np.diff(z)
    dev = np.abs(z - np.median(z))
    return dict(
        sorted=np.sort(z), q10=float(np.quantile(z, 0.10)), q90=float(np.quantile(z, 0.90)),
        q05=float(np.quantile(z, 0.05)), q95=float(np.quantile(z, 0.95)),
        mean=float(z.mean()), std=float(z.std()),
        lev=float(dev.mean()), d_std=float(d.std()) if len(d) > 2 else 1.0,
        d_acf=_acf1(d), d_abs=float(np.abs(d).mean()) if len(d) else 0.0,
        sign=float((z > 0).mean()),
    )


def _two_sample2(h: dict, p: np.ndarray) -> list:
    n = len(p)
    out = []
    out.append(float(np.quantile(p, 0.10) - h["q10"]))
    out.append(float(np.quantile(p, 0.90) - h["q90"]))
    ps = np.sort(p)
    pos = np.searchsorted(h["sorted"], ps, side="right") / len(h["sorted"])
    ecdf = np.arange(1, n + 1) / n
    gap = pos - ecdf
    out.append(float(np.mean(gap ** 2)))
    w = np.clip(pos * (1 - pos), 1e-3, None)
    out.append(float(np.clip(np.mean(gap ** 2 / w), 0, 10)))
    dev = np.abs(p - np.median(p))
    out.append(float(np.log((dev.mean() + 1e-9) / (h["lev"] + 1e-9))))
    d = np.diff(p)
    if len(d) > 2:
        out.append(float(np.log((d.std() + 1e-9) / (h["d_std"] + 1e-9))))
        out.append(_acf1(d) - h["d_acf"])
        out.append(float(np.log((np.abs(d).mean() + 1e-9) / (h["d_abs"] + 1e-9))))
    else:
        out.extend([0.0, 0.0, 0.0])
    for wlen in (10, 25, 100):
        tail = p[-wlen:]
        out.append(float(tail.mean() - h["mean"]))
        out.append(float(np.log((tail.std() + 1e-9) / (h["std"] + 1e-9))))
    out.append(float((p > h["q95"]).mean() - 0.05))
    out.append(float((p < h["q05"]).mean() - 0.05))
    out.append(float((p > 0).mean() - h["sign"]))
    out.append(_acf1(np.sign(p)))
    out.append(_slope_t(p))
    out.append(_slope_t(np.abs(p)))
    return out


SPEC_NFFT = 64
SPEC_BANDS = 8


def _spectrum(v: np.ndarray):
    """Power in eight bands (normalised), spectral entropy, peak position,
    log total power — of the last NFFT points, Hann-windowed."""
    if len(v) < SPEC_NFFT:
        v = np.concatenate([np.zeros(SPEC_NFFT - len(v)), v])
    seg = v[-SPEC_NFFT:] * np.hanning(SPEC_NFFT)
    p = np.abs(np.fft.rfft(seg)) ** 2
    p = p[1:]
    tot = p.sum() + 1e-12
    pn = p / tot
    band = np.add.reduceat(pn, np.linspace(0, len(pn), SPEC_BANDS + 1)[:-1].astype(int))
    ent = float(-(pn * np.log(pn + 1e-12)).sum() / np.log(len(pn)))
    peak = float(np.argmax(pn)) / len(pn)
    return band, ent, peak, float(np.log(tot))


class Spectral:
    """The frequency-domain family: the prefix's spectrum against the
    history's own spectral profile.

    Every other channel this project owns lives in the time domain, and a
    break that rearranges *periodicity* -- the same variance redistributed
    across frequencies -- passes them unseen. The history supplies both the
    mean profile and its spread, so each band's deviation is read in that
    series' own sigmas. Recomputed on the geometric cadence, held between."""

    def __init__(self, history: np.ndarray, norm: "Normalisation") -> None:
        n = len(history)
        z = np.asarray([norm.clip(norm.standardise(float(v), i - n))
                        for i, v in enumerate(history)])
        bands, ents, peaks, tots = [], [], [], []
        stride = max(SPEC_NFFT // 2, 1)
        for i in range(SPEC_NFFT, len(z) + 1, stride):
            b, e, pk, lt = _spectrum(z[i - SPEC_NFFT:i])
            bands.append(b); ents.append(e); peaks.append(pk); tots.append(lt)
        if bands:
            B = np.stack(bands)
            self.hb, self.he = B.mean(0), float(np.mean(ents))
            self.hp, self.ht = float(np.mean(peaks)), float(np.mean(tots))
            self.hb_sd = B.std(0) + 1e-3
            self.he_sd = float(np.std(ents)) + 1e-3
        else:
            b, e, pk, lt = _spectrum(z)
            self.hb, self.he, self.hp, self.ht = b, e, pk, lt
            self.hb_sd, self.he_sd = np.ones(SPEC_BANDS) * 0.1, 0.1
        self.norm = norm
        self.buf = list(z[-SPEC_NFFT:])
        self._next_scan = 1
        self._step = 0
        self.current = [0.0] * 14

    def update(self, x: float) -> list:
        z = self.norm.clip(self.norm.standardise(float(x), self._step))
        self.buf.append(float(z))
        if len(self.buf) > SPEC_NFFT:
            self.buf.pop(0)
        self._step += 1
        if self._step >= self._next_scan:
            self._next_scan = max(self._next_scan + 1, int(self._next_scan * 1.12))
            b, e, pk, lt = _spectrum(np.asarray(self.buf))
            diff = (b - self.hb) / self.hb_sd
            self.current = list(diff) + [
                (e - self.he) / self.he_sd,
                pk - self.hp,
                lt - self.ht,
                float(np.abs(diff).max()),
                float(np.abs(b - self.hb).sum()),
                float(np.dot(b, self.hb) / (np.linalg.norm(b) * np.linalg.norm(self.hb) + 1e-12)),
            ]
        return list(self.current)


class PrefixBattery:
    """The per-prefix battery: 21 statistics per view, raw and asinh, held
    between geometric-cadence recomputes exactly like the retrospective
    scans. The prefix is standardised by the view's own history fit, unclipped
    -- the battery wants to see the tails the detectors are protected from."""

    def __init__(self, history: np.ndarray) -> None:
        history = np.asarray(history, dtype="float64")
        self.views = []
        for transform in (None, np.arcsinh):
            h = history if transform is None else transform(history)
            norm = Normalisation.fit(h)
            n = len(h)
            z_h = np.asarray([norm.standardise(float(v), i - n) for i, v in enumerate(h)])
            self.views.append([norm, _hist_summary(z_h), _hist_summary2(z_h), []])
        self._next_scan = 1
        self._step = 0
        self.current = [0.0] * 82

    def update(self, x: float) -> list:
        for k, view in enumerate(self.views):
            value = float(x) if k == 0 else float(np.arcsinh(x))
            view[3].append(view[0].standardise(value, self._step))
        self._step += 1
        if self._step >= self._next_scan:
            self._next_scan = max(self._next_scan + 1, int(self._next_scan * 1.12))
            v1, v2 = [], []
            for _norm, summary, summary2, prefix in self.views:
                arr = np.asarray(prefix)
                v1.extend(_two_sample(summary, arr))
                v2.extend(_two_sample2(summary2, arr))
            # Column order matches the training matrices: all of v1 (B40),
            # then all of v2 (B2).
            self.current = v1 + v2
        return list(self.current)


class TriMonitor:
    """The dual pipelines of 011 plus the forecaster of 013."""

    def __init__(self, history: np.ndarray, base_forecaster) -> None:
        history = np.asarray(history, dtype="float64")
        self.dual = DualMonitor(history)
        self.forecast = Forecaster(base_forecaster, self.dual.raw.norm, history)
        self.battery = PrefixBattery(history)
        self.spectral = Spectral(history, self.dual.raw.norm)
        # The run-length posterior rides as a six-channel suffix that only the
        # classifier reads (experiment 085: +0.0046 there, nothing elsewhere).
        self.regime = RunLengthMonitor(history)
        # The mass battery is a separate member's input, not a channel of this
        # monitor: appended to the two hundred it loses, trained on its own it
        # disagrees with the ensemble where that pays (114).
        self.mass = MassBattery(history)
        # The second independent member's input: frequency and dependence
        # channels the ensemble does not otherwise read (124-131b).
        self.freqdep = FreqDep(history)
        # The third view: the online window ranked against the history's own
        # windows (140). Read only together with the frequency channels, by
        # the union member, from step UNION_GATE on.
        self.novelty = Novelty(history)
        # The whitened stream (145): AR(p) by BIC, a conditional scale and the
        # innovation ECDF fitted on the history; every test on normal scores.
        # Alone 0.6088 on fold 2 -- nearly the whole ensemble's worth -- and
        # +0.0107 to the blend at a 0.30 share. Read by its own member.
        # With the Shiryaev-Roberts odds of 147 riding along: 111 channels,
        # read by a per-step ranker and a slow classifier (148).
        self.white = WhiteMonitor(history, odds=True)
        self._step = 0

    def update(self, x: float) -> list[float]:
        channels = self.dual.update(float(x))
        z = self.dual.raw.norm.clip(
            self.dual.raw.norm.standardise(float(x), self._step))
        self._step += 1
        return (channels + self.forecast.update(z) + self.battery.update(float(x))
                + self.spectral.update(float(x)) + self.regime.update(float(x))
                + self.mass.update(float(x)) + self.freqdep.update(float(x))
                + self.novelty.update(float(x)) + self.white.update(float(x)))


N_CHANNELS = 2 * (9 + 24 + len(RETRO2_CHANNELS) + 1 + 9) + FORECAST_CHANNELS + 82 + 14 + BOCPD_CHANNELS
NET_CHANNELS = 200   # the trajectory networks read the first two hundred
RANK_CHANNELS = 200  # so do the rankers
CLF_CHANNELS = 206   # the classifier alone reads the run-length suffix
MASS_OFFSET = 206    # the mass battery follows, read only by its own member
FREQDEP_OFFSET = 296 # then the frequency/dependence channels, read only by theirs
NOVELTY_OFFSET = 396 # then the novelty channels; the union member reads 296: as one row
UNION_GATE = 100     # before this step the frequency member alone, after it the union
#: The blend from the gate on: the union's share and the mass member's, the
#: core taking the rest (140c: 0.6251 on fold 2 against 0.6232 for #38).
UNION_MASS, UNION_SHARE = 0.25, 0.25
WHITE_OFFSET = 416   # then the whitened-stream channels with the odds, read only by their member
WHITE2_WIDTH = 111   # ninety whitened channels and twenty-one Shiryaev-Roberts odds
WHITE_SHARE = 0.40   # the member's share of the final blend at every step (148: 0.6405 vs 0.6251)
WHITE_RANK = 0.70    # inside the member: the per-step ranker's share against the classifier's


TCN_DILS = (1, 2, 4, 8, 16, 32)


def _gelu(x: np.ndarray) -> np.ndarray:
    from scipy.special import erf

    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))


class BatchedStreamingChanTCN:
    """Twelve nets, one matrix multiply per step.

    The member weights are stacked along a leading axis, so a step costs one
    einsum per layer for the whole bag instead of twelve python loops --
    profiled at ~1 ms/step for all members together. Verified against the
    per-member forward to 1e-6."""

    def __init__(self, weight_dicts, mu, sd) -> None:
        self.n = len(weight_dicts)
        self.mu = np.asarray(mu, dtype="float64")
        self.sd = np.asarray(sd, dtype="float64")
        st = lambda key: np.stack([np.asarray(w[key], dtype="float64") for w in weight_dicts])
        self.w_inp = st("inp.weight")[:, :, :, 0]
        self.b_inp = st("inp.bias")
        self.conv_w, self.conv_b, self.mix_w, self.mix_b = [], [], [], []
        for li in range(len(TCN_DILS)):
            self.conv_w.append(st(f"blocks.{li}.conv.weight"))
            self.conv_b.append(st(f"blocks.{li}.conv.bias"))
            self.mix_w.append(st(f"blocks.{li}.mix.weight")[:, :, :, 0])
            self.mix_b.append(st(f"blocks.{li}.mix.bias"))
        self.w_head = st("head.weight")[:, 0, :, 0]
        self.b_head = st("head.bias")[:, 0]
        self.hist = [[] for _ in range(len(TCN_DILS) + 1)]

    def update(self, chan_vec) -> np.ndarray:
        x = (np.asarray(chan_vec, dtype="float64") - self.mu) / self.sd
        h = np.einsum("noi,i->no", self.w_inp, x) + self.b_inp
        self.hist[0].append(h)
        for li, d in enumerate(TCN_DILS):
            layer_in = self.hist[li]
            t_idx = len(layer_in) - 1
            acc = self.conv_b[li].copy()
            W = self.conv_w[li]
            for j, off in enumerate((2 * d, d, 0)):
                idx = t_idx - off
                if idx >= 0:
                    acc += np.einsum("nock,nc->no", W[:, :, :, j:j+1], layer_in[idx])[:, :]
            z = _gelu(acc)
            out = layer_in[t_idx] + np.einsum("noc,nc->no", self.mix_w[li], z) + self.mix_b[li]
            self.hist[li + 1].append(out)
        top = self.hist[-1][-1]
        return np.einsum("nc,nc->n", self.w_head, top) + self.b_head


class StreamingChanTCN:
    """The channel-trajectory network, streamed one step at a time.

    The boosted combiners see each step's 186 channels as an isolated
    snapshot; this network sees how they *move* -- ramp shapes, fronts,
    agreement across channels -- through six dilated causal convolutions
    (receptive field 127 steps). Trained with the per-step ranking loss; here
    only the numpy forward runs, incrementally: each step costs one new
    position per layer, verified against the torch forward to 2e-7."""

    def __init__(self, weights: dict, mu, sd) -> None:
        self.w = {k: np.asarray(v, dtype="float64") for k, v in weights.items()}
        self.mu = np.asarray(mu, dtype="float64")
        self.sd = np.asarray(sd, dtype="float64")
        self.hist = [[] for _ in range(len(TCN_DILS) + 1)]

    def update(self, chan_vec) -> float:
        x = (np.asarray(chan_vec, dtype="float64") - self.mu) / self.sd
        h = self.w["inp.weight"][:, :, 0] @ x + self.w["inp.bias"]
        self.hist[0].append(h)
        for li, d in enumerate(TCN_DILS):
            layer_in = self.hist[li]
            t_idx = len(layer_in) - 1
            W = self.w[f"blocks.{li}.conv.weight"]
            acc = self.w[f"blocks.{li}.conv.bias"].copy()
            for j, off in enumerate((2 * d, d, 0)):
                idx = t_idx - off
                if idx >= 0:
                    acc += W[:, :, j] @ layer_in[idx]
            z = _gelu(acc)
            out = (layer_in[t_idx]
                   + self.w[f"blocks.{li}.mix.weight"][:, :, 0] @ z
                   + self.w[f"blocks.{li}.mix.bias"])
            self.hist[li + 1].append(out)
        top = self.hist[-1][-1]
        return float(self.w["head.weight"][0, :, 0] @ top + self.w["head.bias"][0])


def train(
    datasets: List[Tuple[int, List[float], List[float], Optional[int]]],
    model_directory_path: str,
) -> None:
    """Full rebuild: pretrain the forecaster on every history, then walk every
    series through the TriMonitor and fit the combiner on all 104 channels."""
    import lightgbm as lgb

    feats_all, targ_all = [], []
    for _sid, x_hist, _x_online, _tau in datasets:
        history = np.asarray(x_hist, dtype="float64")
        norm = Normalisation.fit(history)
        n = len(history)
        z = np.asarray([norm.clip(norm.standardise(float(v), i - n))
                        for i, v in enumerate(history)])
        d = Forecaster._deviations(z)
        idx = np.arange(FORECAST_LAGS + FORECAST_K, n)
        if len(idx) > 8:
            feats_all.append(_deviation_features(d, idx)[::2])
            targ_all.append(d[idx][::2])
    base = lgb.train(
        dict(objective="l2", learning_rate=0.05, num_leaves=15,
             min_data_in_leaf=200, feature_fraction=0.9, bagging_fraction=0.8,
             bagging_freq=1, verbose=-1, deterministic=True,
             force_row_wise=True, num_threads=4, seed=0),
        lgb.Dataset(np.vstack(feats_all).astype("float32"),
                    np.concatenate(targ_all).astype("float32")),
        num_boost_round=200)

    rows: list[list[float]] = []
    targets: list[int] = []
    steps: list[int] = []
    for _sid, x_hist, x_online, tau in datasets:
        online = np.asarray(x_online, dtype="float64")
        if len(online) == 0:
            continue
        monitor = TriMonitor(np.asarray(x_hist, dtype="float64"), base)
        for step, value in enumerate(online):
            rows.append(monitor.update(float(value)))
            targets.append(int(tau is not None and step >= tau))
            steps.append(step)

    x = np.asarray(rows, dtype="float32")
    y = np.asarray(targets, dtype="int64")
    step_array = np.asarray(steps, dtype="int64")
    unique, counts = np.unique(step_array, return_counts=True)
    lookup = dict(zip(unique.tolist(), counts.tolist()))
    weights = np.array([1.0 / lookup[s] for s in step_array.tolist()], dtype="float64")
    weights *= len(weights) / weights.sum()

    booster = lgb.LGBMClassifier(
        objective="binary",
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=500,
        subsample=0.8,
        subsample_freq=1,
        # Half the columns per tree: with 186 channels the strongest
        # decorrelator the resweep found -- each tree sees a different half
        # of the ensemble's eyes.
        colsample_bytree=0.5,
        reg_lambda=10.0,
        n_estimators=300,
        random_state=0,
        n_jobs=4,
        deterministic=True,
        force_row_wise=True,
        verbose=-1,
    )
    booster.fit(x, y, sample_weight=weights)

    # The ranking half of the blend: the same rows sorted by step, every
    # cross-section a group, trained to order it. Truncation 2000 is the
    # measured optimum (500 cuts the signal, 8000 dilutes the gradient).
    order = np.argsort(step_array, kind="stable")
    _, group_sizes = np.unique(step_array[order], return_counts=True)
    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=500,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=10.0,
        n_estimators=300,
        random_state=0,
        n_jobs=4,
        deterministic=True,
        force_row_wise=True,
        verbose=-1,
        lambdarank_truncation_level=2000,
        label_gain=[0, 1],
    )
    ranker.fit(x[order], y[order], group=group_sizes)

    joblib.dump(
        {"booster": booster, "rankers": [ranker], "forecaster": base.model_to_string()},
        os.path.join(model_directory_path, "model.joblib"),
    )


def infer(
    datasets: Iterable[Tuple[List[float], Iterable[float]]],
    model_directory_path: str,
):
    import lightgbm as lgb

    model = joblib.load(os.path.join(model_directory_path, "model.joblib"))
    # Raw boosters with a single thread per call: the sklearn wrappers spawn
    # a thread pool on every one-row predict, and five million spawns across
    # eight workers deadlocked run #109121 into a 7-hour timeout (exit 124).
    clf_booster = model["booster"].booster_
    mass_booster = model["mass_classifier"].booster_
    freqdep_booster = model["freqdep_classifier"].booster_
    union_booster = model["union_classifier"].booster_
    white_booster = model["white2_classifier"].booster_
    white_ranker = model["white2_ranker"].booster_
    rank_boosters = [r.booster_ for r in model["rankers"]]
    base_forecaster = lgb.Booster(model_str=model["forecaster"])
    net_list = model.get("nets") or []
    net_mu, net_sd = model.get("net_mu"), model.get("net_sd")

    yield  # Signal readiness to the runner.

    for x_historical, x_online in datasets:
        monitor = TriMonitor(np.asarray(x_historical, dtype="float64"), base_forecaster)
        # The triple: ranker 0.36, classifier 0.24, channel-trajectory net
        # 0.40 -- fold-0 0.6068 against 0.6045 for the pair. The net is the
        # first genuinely unlike member: it reads the channels' motion, which
        # per-step trees cannot see.
        # The heavy-member A/B: the exact #19 weights (0.35 bagged rankers +
        # 0.15 classifier + 0.50 nets), with the four fold-nets replaced by
        # eight members trained on 88% of ALL data each, 16 epochs,
        # best-epoch by a private per-member holdout. Any cloud delta
        # against #19's 0.5877 is member quality and nothing else.
        bag_net = BatchedStreamingChanTCN(net_list, net_mu, net_sd) if net_list else None
        for step, point in enumerate(x_online):
            channels = np.asarray(monitor.update(float(point)), dtype="float64")
            clf_row = channels[:CLF_CHANNELS].reshape(1, -1)
            rank_row = channels[:RANK_CHANNELS].reshape(1, -1)
            net_row = channels[:NET_CHANNELS]
            mass_row = channels[MASS_OFFSET:FREQDEP_OFFSET].reshape(1, -1)
            mass_prob = float(mass_booster.predict(mass_row, num_threads=1)[0])
            prob = float(clf_booster.predict(clf_row, num_threads=1)[0])
            bag = sum(1.0 / (1.0 + math.exp(-float(rb.predict(rank_row, num_threads=1)[0])))
                      for rb in rank_boosters) / len(rank_boosters)
            trees = 0.7 * bag + 0.3 * prob
            if bag_net is None:
                yield trees
            else:
                # Twelve members: #28's six and six more under the last-epoch
                # rule (082b). Either six reads 0.616 on fold 2 in this blend;
                # together they read the same with half the member variance.
                logits = bag_net.update(net_row)
                net_sig = float(np.mean(1.0 / (1.0 + np.exp(-logits))))
                # Twenty-four networks here, twelve in #36: fold 2 cannot
                # separate them, the cloud reads this recipe within 0.001, and
                # halved member variance is the only thing a larger pool buys.
                # The ensemble as shipped, then a quarter weight on the mass
                # member: fold 2 rises from 0.6167 to 0.6205 at this share, and
                # the gain grows monotonically from 0.10 to 0.30 (114).
                # Two independent members on top of the core: the mass member
                # at a quarter (#36: +0.0039 in the cloud) and the frequency/
                # dependence member at a fifth (131b: +0.0025 on fold 2 at
                # correlation 0.37). Jointly tuned: fold 2 reads 0.6232 on a
                # flat plateau across 0.20-0.30 / 0.15-0.25.
                core = 0.45 * trees + 0.55 * net_sig
                white_row = channels[WHITE_OFFSET:].reshape(1, -1)
                white_clf = float(white_booster.predict(white_row, num_threads=1)[0])
                white_rank = 1.0 / (1.0 + math.exp(-float(white_ranker.predict(white_row, num_threads=1)[0])))
                # The member: a per-step ranker (0.6216 alone on fold 2) and a
                # slow classifier (0.6186), Spearman 0.61 between them, blended
                # 0.7 / 0.3 (148).
                white_prob = WHITE_RANK * white_rank + (1.0 - WHITE_RANK) * white_clf
                if step < UNION_GATE:
                    # #38 as shipped: the novelty windows are unfilled this
                    # early and the union reads noise there (140: -0.005 on
                    # steps 0-100), so the frequency member stands alone.
                    freqdep_row = channels[FREQDEP_OFFSET:NOVELTY_OFFSET].reshape(1, -1)
                    freqdep_prob = float(freqdep_booster.predict(freqdep_row, num_threads=1)[0])
                    blend = 0.55 * core + 0.25 * mass_prob + 0.20 * freqdep_prob
                else:
                    # From the gate on, the frequency channels and the
                    # novelty channels are read together by one classifier:
                    # a member that agrees with the frequency member at 0.21
                    # and lifts fold 2 from 0.6232 to 0.6251 (140c).
                    union_row = channels[FREQDEP_OFFSET:WHITE_OFFSET].reshape(1, -1)
                    union_prob = float(union_booster.predict(union_row, num_threads=1)[0])
                    blend = ((1.0 - UNION_MASS - UNION_SHARE) * core
                             + UNION_MASS * mass_prob + UNION_SHARE * union_prob)
                # #39 as shipped (cloud 0.6056), then the whitened member at
                # 0.40 at every step. #41 carried its classifier alone at 0.30
                # (fold 2 0.6369, cloud 0.6186); with the odds and the ranker
                # fold 2 reads 0.6405.
                yield (1.0 - WHITE_SHARE) * blend + WHITE_SHARE * white_prob
