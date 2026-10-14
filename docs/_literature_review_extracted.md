[1] (Heading 1) 1.2 国内外研究现状

[2] (Normal) 地铁物流配送转运点布局优化并非单一的物流设施选址问题，而是地铁网络结构、站点客流组织、城市配送需求、末端接驳能力、站点空间容量与客货协同组织共同作用下的复合型优化问题。近年来，城市轨道交通研究逐步从静态拓扑分析转向客流加权、扰动传播、韧性恢复和资源配置；城市物流节点选址研究也由传统配送中心布局扩展到微枢纽、智能快递柜、两阶段配送和绿色低碳配送；与此同时，地铁物流网络系统、客货协同运输和城市轨道交通资源复合利用逐渐成为城市物流研究的重要方向。总体来看，既有研究为本文提供了复杂网络评价、设施选址优化和地铁物流网络规划等理论基础，但在地铁站点重要度评价结果与物流转运点布局模型的衔接、客流干扰与站点容量约束刻画、地铁干线运输与末端接驳配送协同优化等方面仍有进一步研究空间。

[3] (Heading 2) 1.2.1 地铁网络站点重要度评价研究

[4] (Normal) 地铁网络站点重要度评价主要回答“哪些站点在网络运行中更关键”这一问题。现有研究大体经历了由静态拓扑评价向客流加权评价、由关键节点识别向脆弱性与韧性分析、由单一排序结果向恢复优化和资源配置决策转化的演进过程。早期复杂网络方法多以站点为节点、区间或换乘关系为边，利用度中心性、介数中心性、接近中心性和网络效率等指标识别关键站点。近年来，随着城市轨道交通网络规模扩大和运营扰动增多，单纯依赖拓扑指标已难以反映真实运营状态，研究重点逐步转向结构、客流、扰动和恢复能力的综合评价。Wei等系统梳理了城市轨道交通韧性研究的概念、评价方法与发展趋势，指出轨道交通网络评价应从结构可靠性扩展到预防、吸收、恢复和适应全过程[1]；Hu等进一步总结了城市轨道交通网络韧性评估与恢复策略，强调关键站点识别应服务于恢复顺序优化和运行保障决策[2]。

[5] (Normal) 围绕关键站点识别问题，近年研究逐渐突破传统中心性指标局限，开始将站点失效后的替代连接能力、服务损失和恢复优先级纳入评价框架。Kopsidas和Kepaptsoglou提出替代复杂网络分析方法识别地铁系统关键站点，将站点失效后的网络替代能力纳入评价框架[3]；Zhang等基于综合VoteRank算法识别城市轨道交通关键站点，并通过鲁棒性分析验证关键站点失效对网络性能的影响[4]。这类研究表明，站点重要度评价已由“节点在网络中是否居中”转向“节点失效后会造成多大功能损失”。在更近的研究中，Dynamic DomiRank、Flow DomiGCN、跨层融合和级联失效稳定性分析等方法被引入关键站点识别，进一步强化了动态图学习和多层网络视角[5-8]。

[6] (Normal) 以上海、深圳、南京、北京、重庆等大城市地铁网络为对象的案例研究，进一步将关键站点识别与高密度客流、复杂换乘和突发扰动情景相结合，突出评价结果在运营组织和应急恢复中的工程应用价值。Xu和Xu以上海地铁为例，采用PCA-TPE方法识别关键站点，将多维指标压缩融合后形成综合评价结果[9]；Meng等基于深圳地铁高峰期大规模出行数据构建客流加权网络，说明加权网络比无权拓扑网络更能反映站点真实运营作用[10]；Liu等从出行行为角度评价城市轨道交通网络脆弱性，揭示乘客路径选择和换乘行为对站点失效后服务损失的影响[11]。围绕扰动恢复，Chen等面向短期运营扰动评估城市轨道交通网络韧性[12]，Tan等研究有限预算下北京地铁脆弱站点恢复序列优化[13]，Jiao等识别了暴雨灾害下城市轨道交通站点韧性的关键影响因素[14]。进一步地，Jiao等以重庆地铁为例，采用云模型开展站点暴雨灾害韧性评价[15]；Meng等从2004—2021年网络演化视角分析深圳地铁复杂拓扑和节点重要度变化[16]，Tian等基于客流加权网络评价城市轨道交通网络演化过程中的动态韧性[17]。这些研究表明，城市轨道交通站点重要度评价已由单一节点排序逐步转向客流加权、灾害韧性、动态演化和恢复决策。

[7] (Normal) 从方法脉络看，地铁站点重要度评价可概括为“拓扑中心性—客流加权—韧性恢复—物流适配”四个递进层次。第一类拓扑中心性方法强调站点在网络连接结构中的桥接和控制作用，能够识别换乘性强、连通性高的结构关键节点；第二类客流加权方法进一步引入进出站客流、换乘客流和OD需求，弥补单纯拓扑指标难以反映真实运营负荷的不足；第三类脆弱性与韧性评价方法关注节点失效、线路中断、灾害扰动和恢复过程，将关键站点识别延伸至运营保障和资源配置决策；第四类面向物流适配的评价则需要在上述基础上进一步考察站点空间容量、客货流线冲突、装卸组织条件和末端接驳可达性。由此可见，站点在客运网络中的“重要”并不等同于其适合作为地铁物流配送转运点，二者之间需要经过物流功能适配性筛选和约束转化。总体而言，现有研究为本文识别地铁物流潜在转运站点提供了复杂网络评价和韧性分析基础，但其评价目标主要服务于客运网络运行安全、扰动恢复和保护资源配置，尚未充分面向物流功能嵌入场景。对于地铁物流配送转运点布局而言，部分换乘枢纽虽然拓扑重要度和客流重要度较高，但可能存在站厅站台空间紧张、客货流线交叉、装卸作业窗口受限和客运组织风险较高等问题；相反，部分客流压力较小且邻近物流需求集聚区的普通站点，可能更适合承担物流接驳和转运功能。因此，本文在借鉴站点重要度评价方法时，需要将拓扑重要度、客流影响、站点容量和末端接驳条件转化为候选站点筛选依据，而非简单以关键站点排序结果替代转运点布局决策。

[8] (Heading 2) 1.2.2 城市物流节点选址模型与方法研究

[9] (Normal) 城市物流节点选址主要解决“物流节点布设在哪里、服务哪些需求以及如何组织末端接驳配送”的问题。随着电子商务、即时零售、社区配送和低碳交通的发展，城市物流节点已由传统大型配送中心扩展为城市微枢纽、智能快递柜、自提点、移动仓和多式联运枢纽。相关研究也由经典设施选址逐步演化为选址—路径联合优化、多目标决策、协同配送和数据驱动规划。Boysen等从运筹优化视角综述最后一公里配送概念，指出末端接驳配送节点已经与车辆路径、客户取货方式、自动化配送设施和众包配送深度耦合[18]；Taniguchi等从智慧城市与可持续城市货运角度指出，城市物流节点布局不仅是企业成本问题，也涉及道路资源、环境影响和城市治理目标[19]。

[10] (Normal) 围绕城市物流节点选址，近年文献主要形成了“传统设施选址—两阶段配送—微枢纽布局—多准则决策”逐步深化的研究脉络。第一类研究关注两阶段配送和选址—路径联合优化。Enthoven等将货运自行车和包裹柜纳入两阶段车辆路径问题，说明节点位置、服务覆盖和二级配送路径需要协同优化[20]；Yu等考虑可循环快递包装，建立设施选址与车辆路径同步优化模型[21]；Hajghani等构建考虑可持续性和不确定性的两级位置—路径模型，将经济、环境和需求波动纳入统一框架[22]。第二类研究关注微枢纽和末端设施布局。Lagorio和Pinto通过案例分析界定城市货运微枢纽[23]；Novotna等围绕可持续末端配送中的微枢纽选址问题，耦合BWM、CRITIC和WASPAS等多准则决策方法，对候选节点进行综合排序，为高密度城区末端物流节点筛选提供了方法参考[24]；Castillo等以巴塞罗那为例研究微枢纽分布与绿色路径优化[25]。上述研究表明，城市物流节点正呈现小型化、近端化和低碳化趋势，节点选址也由单纯追求成本节约转向兼顾服务覆盖、空间适配和绿色配送绩效。第三类研究关注多准则评价和新型末端接驳配送设施。Janinhoff等综述户外自提配送[26]，Sawik等通过仿真—优化方法评价自动包裹柜使用方案[27]，Ranjbari等基于实地数据检验包裹柜对配送时间的影响[28]，表明末端设施绩效不仅取决于空间位置，还受到用户行为、设施利用率和运营组织影响。

[11] (Normal) 在模型方法上，近五年研究进一步强调多主体协同、多式联运和动态运行约束。Yıldız等从战略与战术一体化角度研究多式联运枢纽网络设计，将枢纽选址、路径组织和线路能力配置纳入统一模型[29]；Aloui等对协同可持续运输进行系统综述，指出城市物流节点布局需要处理多主体协作、资源共享和环境目标之间的关系[30]；Zhen等研究带时间窗和释放时间的多仓多行程车辆路径问题，揭示节点布局与配送批次、时间窗和车辆调度之间的耦合关系[31]；Engesser等综述自动配送解决方案，说明无人配送技术将进一步改变末端节点选址逻辑[32]。这些研究共同表明，城市物流节点选址已从静态设施位置选择转向面向运行过程的系统优化。

[12] (Normal) 同时，多准则评价方法被广泛用于城市货运方案比选与候选节点排序。Alvarez Gallo和Maheut系统梳理了城市货运物流方案评价中的多准则分析方法，指出评价指标通常涉及经济成本、环境影响、服务水平和实施可行性等维度[33]；Yildiz等采用层次分析法开展城市枢纽选址研究，为城市物流节点选址中的指标权重确定和候选方案排序提供了方法参考[34]。

[13] (Normal) 面向高密度城市空间的案例研究进一步强调配送需求集聚、道路资源约束和政策适配。Lyu等利用离散事件仿真和聚类分析构建随机两级枢纽—辐射城市货运物流架构，体现了城市物流节点布局由经验选址向数据驱动决策转变[35]；Aloui等构建协同策略下可持续城市货运运输的集成优化方法，为城市配送节点与协同运输组织联合优化提供参考[36]。从研究侧重点看，面向具体城市的应用研究通常更关注人口密度、商业设施、道路可达性、配送半径、车辆通行限制和低碳政策等现实约束；偏方法建模的研究则更重视模型抽象、算法设计和多场景推广。二者共同推动了城市物流节点选址由“成本最小化”向“成本—效率—环境—公平—协同”综合优化转变。

[14] (Normal) 尽管城市物流节点选址研究已经较为丰富，但直接应用于地铁物流配送转运点布局仍存在适配不足。第一，现有研究多以道路配送网络为基础，虽然涉及微枢纽、快递柜和两阶段配送，但较少将地铁站点、轨道交通网络和地下空间作为候选物流节点进行系统建模。第二，城市物流节点选址通常强调需求覆盖、运输成本和末端路径效率，而地铁物流配送转运点还必须考虑客运安全、站点容量、列车时刻和客货流线冲突。第三，既有模型较少将交通网络节点重要度评价结果嵌入选址模型，导致候选节点筛选主要依赖需求密度或道路可达性，难以反映地铁网络结构和站点功能差异。因此，本文需要在城市物流节点选址模型基础上，引入地铁站点重要度、分时客流容量和轨道运输特征，构建面向地铁物流场景的转运点布局优化模型。

[15] (Heading 2) 1.2.3 地铁物流配送转运点布局优化研究

[16] (Normal) 地铁物流配送转运点布局优化是城市物流绿色化、轨道交通资源复合利用和地下空间开发交叉形成的新兴方向。与一般物流节点不同，地铁物流配送转运点同时具有货物集散节点和客运交通节点双重属性，其布局不仅取决于物流需求和配送成本，还受地铁线路结构、列车运能、站点空间、客流组织、安检装卸流程和客货协同安全约束。近年来，相关研究大体经历了从可行性论证到网络规划、从单点选址到地上—地下联动、从静态布局到客货协同组织与韧性评价的发展过程。

[17] (Normal) 在网络规划层面，地铁物流研究已形成较为明确的系统设计路径。Zhao等基于枢纽—辐射网络模型优化城市内部地铁物流系统设计，证明地铁网络可作为城市配送干线或次干线运输通道，从而降低道路配送压力并提升系统效率[37]；Sun等以南京为案例构建地铁物流网络系统网络规划多目标优化模型，将设施选址、网络连接和服务需求纳入统一框架[38]；Hu等面向混合不确定性提出多目标协同进化优化方法，说明地铁物流网络设计需要同时应对需求波动、成本变化和系统容量不确定性[39]。Gong等对地铁物流网络系统的可持续设计与运营管理进行主题综述，指出该领域正从概念论证转向系统规划、运营组织和政策实施[40]；Zhang等基于科学计量方法分析公共交通视角下地下物流的发展方向，进一步说明既有城市轨道交通资源复合利用正在成为地下物流研究的重要趋势[41]。

[18] (Normal) 在转运点布局与末端接驳配送衔接方面，研究重点逐步由单一线路配送转向地上—地下联动网络设计。Zheng等研究基于单条地铁线路的城市物流配送路径规划，说明地铁线路可与地面配送车辆共同构成城市配送链条[42]；Zhang等提出双层城市地铁地上—地下物流配送节点选址模型，将地下地铁运输与地面配送节点联合考虑[43]；Bai等研究可持续城市货运配送中的地铁物流枢纽选址，强调转运点布局需要兼顾物流效率、服务覆盖和可持续性[44]；Tian等提出面向地铁型地下物流系统的两阶段站点—网络设计框架，先对候选站点适配性进行评价，再构建试点网络方案，体现了“站点筛选—网络构建—运营优化”的连续决策逻辑[45]。近期，Yang等从供需关系角度比较地铁物流系统不同布局策略[46]，Hu和Guo研究不完全枢纽选址—路径网络设计[47]，说明地铁物流布局研究正由单纯节点选择向供需匹配、网络结构设计和选址—路径一体化方向深化。

[19] (Normal) 在客货协同组织层面，地铁物流研究逐步拓展至利益机制、列车服务、容量分配和站内功能组织。Ma等从博弈论角度分析地铁融合城市物流系统，指出轨道运营企业、物流企业和管理部门之间的策略互动会影响系统实施效果[48]；Ye等提出客货混运地铁货运方案，探讨在既有客运系统中组织货运服务的可行路径[49]；Yang等采用Wasserstein分布鲁棒优化方法研究地铁物流网络系统中的列车运行与货物分配问题[50]；Li等进一步研究列车服务与货物配送联合优化，将列车服务计划、货物配送需求和系统运行成本纳入统一优化框架，说明转运点布局不能脱离列车运行组织和货物流分配[51]。在站内空间层面，Hu等针对客货一体化地铁站内物流功能分区和设备配置提出双层规划方法[52]；Du等研究地下物流系统定价与容量优化[53]；Li等研究地铁—公路协同配送中的拆分需求路径规划[54]。这些研究表明，地铁物流配送转运点布局已从“选哪些站点”扩展到“如何分配容量、如何组织列车、如何衔接地面配送”的综合优化问题。

[20] (Normal) 在应急韧性和低碳治理场景中，相关研究将地铁系统视为城市配送干线或次干线运输资源，重点讨论其在缓解道路拥堵、降低碳排放和提升公共卫生事件下配送韧性等方面的作用。Hou等以上海为案例评估公共卫生事件中地铁物流网络系统维持城市物流绩效的作用[55]；Sun等将地铁物流系统纳入公共卫生事件下医疗物资配送路径优化[56]；Li等基于多层相依网络对地铁型城市地下物流系统韧性进行建模，揭示客运网络、物流网络和需求节点之间的相互依赖关系[57]。Derse和Van Woensel综述人货一体化运输研究，指出公共交通资源与城市货运共享需要处理容量分配、运营组织、安全责任和商业模式等问题[58]。此外，Hu和Wang从设计与可持续评价角度研究地铁物流网络系统[59]，Sun等从低碳收益视角构建地铁货运系统利益相关者三方演化博弈模型[60]，进一步拓展了地铁物流研究的政策激励和可持续治理视角。

[21] (Normal) 总体而言，地铁物流配送转运点布局研究已形成“网络规划—转运点布局—客货协同—运营韧性”的基本脉络，但仍存在三类关键不足。第一，站点重要度评价与转运点选址模型衔接不足。现有研究虽然关注地铁物流网络设计，但候选转运点筛选多基于需求覆盖和运输成本，较少将站点拓扑重要度、客流加权重要度和失效影响转化为候选站点适配性评价指标。针对这一不足，本文拟将地铁网络站点重要度评价结果引入候选站点筛选过程，并结合物流需求分布和末端服务半径识别具备物流转运潜力的站点。第二，客流干扰和站点容量约束刻画不足。许多模型将地铁站点作为普通物流设施处理，对分时客流、站厅站台空间、装卸作业时间和客货流线冲突考虑不够，可能导致选址方案虽具物流效率但工程实施性不足。针对这一不足，本文拟在模型中设置站点处理能力、负载率、最大换乘次数和线路转运点数量等约束，以反映客运安全和站点作业能力对物流功能嵌入的限制。第三，转运点布局、需求分配、轨道运输和末端接驳配送之间的一体化程度仍需加强。已有研究常分别优化网络规划、列车运行或末端路径，尚未充分揭示建设成本、服务覆盖、站点容量、碳减排和客运干扰之间的多目标权衡。针对这一不足，本文拟构建综合成本最小化与地铁服务比例最大化的选址—分配双目标模型，统筹地铁干线运输、物流需求分配和末端接驳配送，为城市轨道交通资源复合利用和绿色城市配送体系建设提供方法支撑。

[22] (Heading 2) 参考文献

[23] (Normal) [1] WEI Y, YANG X, XIAO X, et al. Understanding the resilience of urban rail transit: concepts, reviews, and trends[J]. Engineering, 2024, 41: 7-18.

[24] (Normal) [2] HU J, YANG M, ZHEN Y. A review of resilience assessment and recovery strategies of urban rail transit networks[J]. Sustainability, 2024, 16(15): 6390.

[25] (Normal) [3] KOPSIDAS A, KEPAPTSOGLOU K. Identification of critical stations in a metro system: a substitute complex network analysis[J]. Physica A: Statistical Mechanics and its Applications, 2022, 596: 127123.

[26] (Normal) [4] ZHANG J, ZHOU Y, WANG S, et al. Critical station identification and robustness analysis of urban rail transit networks based on comprehensive vote-rank algorithm[J]. Chaos, Solitons & Fractals, 2024, 178: 114379.

[27] (Normal) [5] LI J, LU Q C, XU P C, et al. Critical station identification for cascading failure mitigation considering the Lyapunov-stability of metro stations[J]. Reliability Engineering & System Safety, 2025, 256: 110772.

[28] (Normal) [6] ZHANG J, LI W, LI F, et al. Critical station identification and vulnerability assessment of metro networks based on dynamic DomiRank and Flow DomiGCN[J]. Sustainability, 2026, 18(4): 1781.

[29] (Normal) [7] CHEN J, LIU X, DU B, et al. Vulnerability reduction of an urban rail transit network by allocating protective resources: a scenario model approach[J]. Transportation Research Record, 2025, 2679(9): 751-768.

[30] (Normal) [8] ZHANG Y, SHUAI B, LI S, et al. Critical station identification framework combining layered computation with cross-layer fusion for urban public transport networks[J]. Reliability Engineering & System Safety, 2026, 274: 112470.

[31] (Normal) [9] XU X, XU C. Recognizing critical stations in urban rail transit networks based on the PCA-TPE method: Shanghai metro as an example[J]. Journal of Advanced Transportation, 2024, 2024(1): 1129799.

[32] (Normal) [10] MENG Y, ZHAO X, LIU J, et al. Data-driven complexity analysis of weighted Shenzhen metro network based on urban massive mobility in the rush hours[J]. Physica A: Statistical Mechanics and its Applications, 2023, 610: 128403.

[33] (Normal) [11] LIU B, ZHU G, LI X, et al. Vulnerability assessment of the urban rail transit network based on travel behavior analysis[J]. IEEE Access, 2021, 9: 1407-1419.

[34] (Normal) [12] CHEN J, LIU J, DU B, et al. Resilience assessment of an urban rail transit network under short-term operational disturbances[J]. IEEE Transactions on Intelligent Transportation Systems, 2022, 23(12): 24841-24853.

[35] (Normal) [13] TAN E, LIU B, GUO C, et al. Restoration sequence optimization for vulnerable metro stations with limited budget: a case study of Beijing, China[J]. Physica A: Statistical Mechanics and its Applications, 2024, 653: 130102.

[36] (Normal) [14] JIAO L, LI D, ZHANG Y, et al. Identification of the key influencing factors of urban rail transit station resilience against disasters caused by rainstorms[J]. Land, 2021, 10(12): 1298.

[37] (Normal) [15] JIAO L, ZHU Y, HUO X, et al. Resilience assessment of metro stations against rainstorm disaster based on cloud model: a case study in Chongqing, China[J]. Natural Hazards, 2023, 116(2): 2311-2337.

[38] (Normal) [16] MENG Y, QI Q, LIU J, et al. Dynamic evolution analysis of complex topology and node importance in Shenzhen metro network from 2004 to 2021[J]. Sustainability, 2022, 14(12): 7234.

[39] (Normal) [17] TIAN T, LIANG Y, PENG Z, et al. Assessing the dynamic resilience of urban rail transit networks during their evolution using a ridership-weighted network[J]. PLOS ONE, 2023, 18(9): e0291639.

[40] (Normal) [18] BOYSEN N, FEDTKE S, SCHWERDFEGER S. Last-mile delivery concepts: a survey from an operational research perspective[J]. OR Spectrum, 2021, 43(1): 1-58.

[41] (Normal) [19] TANIGUCHI E, THOMPSON R G, YAMADA T. Smart city for sustainable urban freight logistics[J]. International Journal of Production Research, 2021, 59(7): 2079-2089.

[42] (Normal) [20] ENTHOVEN D L J U, JARGALSAIKHAN B, ROODBERGEN K J, et al. The two-echelon vehicle routing problem with covering options: city logistics with cargo bikes and parcel lockers[J]. Computers & Operations Research, 2020, 118: 104919.

[43] (Normal) [21] YU H, ZHANG J, CHEANG B, et al. A simultaneous facility location and vehicle routing problem with recyclable express packaging consideration for sustainable city logistics[J]. Sustainable Cities and Society, 2023, 98: 104857.

[44] (Normal) [22] HAJGHANI M, FORGHANI M A, HEIDARI A, et al. A two-echelon location routing problem considering sustainability and hybrid open and closed routes under uncertainty[J]. Heliyon, 2023, 9(3): e14258.

[45] (Normal) [23] LAGORIO A, PINTO R. Defining urban freight microhubs: a case study analysis[J]. Sustainability, 2022, 14(1): 532.

[46] (Normal) [24] NOVOTNA M, SVADLENKA L, JOVCIC S, et al. Micro-hub location selection for sustainable last-mile delivery[J]. PLOS ONE, 2022, 17(7): e0270926.

[47] (Normal) [25] CASTILLO C, PANADERO J, ALVAREZ-PALAU E J, et al. Towards greener city logistics: an application of agile routing algorithms to optimize the distribution of micro-hubs in Barcelona[J]. European Transport Research Review, 2024, 16(1): 25.

[48] (Normal) [26] JANINHOFF L, KLEIN R, SAILER D, et al. Out-of-home delivery in last-mile logistics: a review[J]. Computers & Operations Research, 2024, 168: 106686.

[49] (Normal) [27] SAWIK B, SERRANO-HERNANDEZ A, MURO A, et al. Multi-criteria simulation-optimization analysis of usage of automated parcel lockers: a practical approach[J]. Mathematics, 2022, 10(23): 4423.

[50] (Normal) [28] RANJBARI A, DIEHL C, DALLA CHIARA G, et al. Do parcel lockers reduce delivery times Evidence from the field[J]. Transportation Research Part E: Logistics and Transportation Review, 2023, 172: 103070.

[51] (Normal) [29] YILDIZ B, YAMAN H, KARASAN O E. Hub location, routing, and route dimensioning: strategic and tactical intermodal transportation hub network design[J]. Transportation Science, 2021, 55(6): 1351-1369.

[52] (Normal) [30] ALOUI A, HAMANI N, DERROUICHE R, et al. Systematic literature review on collaborative sustainable transportation: overview, analysis and perspectives[J]. Transportation Research Interdisciplinary Perspectives, 2021, 9: 100291.

[53] (Normal) [31] ZHEN L, MA C, WANG K, et al. Multi-depot multi-trip vehicle routing problem with time windows and release dates[J]. Transportation Research Part E: Logistics and Transportation Review, 2020, 135: 101866.

[54] (Normal) [32] ENGESSER V, ROMBAUT E, VANHAVERBEKE L, et al. Autonomous delivery solutions for last-mile logistics operations: a literature review and research agenda[J]. Sustainability, 2023, 15(3): 2774.

[55] (Normal) [33] ALVAREZ GALLO S, MAHEUT J. Multi-criteria analysis for the evaluation of urban freight logistics solutions: a systematic literature review[J]. Mathematics, 2023, 11(19): 4089.

[56] (Normal) [34] YILDIZ B, SCERBACOVA N, BALL C. Analytic hierarchy process for city hub location selection: the Viennese case[J]. Transportation Research Procedia, 2020, 46: 77-84.

[57] (Normal) [35] LYU Z, PONS D, PALLIPARAMPIL G, et al. Optimising urban freight logistics using discrete-event simulation and cluster analysis: a stochastic two-tier hub-and-spoke architecture approach[J]. Smart Cities, 2023, 6(5): 2347-2366.

[58] (Normal) [36] ALOUI A, HAMANI N, DELAHOCHE L. An integrated optimization approach using a collaborative strategy for sustainable cities freight transportation: a case study[J]. Sustainable Cities and Society, 2021, 75: 103331.

[59] (Normal) [37] ZHAO L, ZHOU J, LI H, et al. Optimizing the design of an intra-city metro logistics system based on a hub-and-spoke network model[J]. Tunnelling and Underground Space Technology, 2021, 116: 104086.

[60] (Normal) [38] SUN X, HU W, XUE X, et al. Multi-objective optimization model for planning metro-based underground logistics system network: Nanjing case study[J]. Journal of Industrial and Management Optimization, 2023, 19(1): 170.

[61] (Normal) [39] HU W, DONG J, YANG K, et al. Network planning of metro-based underground logistics system against mixed uncertainties: a multi-objective cooperative co-evolutionary optimization approach[J]. Expert Systems with Applications, 2023, 217: 119554.

[62] (Normal) [40] GONG D, TIAN J, HU W, et al. Sustainable design and operations management of metro-based underground logistics systems: a thematic literature review[J]. Buildings, 2023, 13(8): 1888.

[63] (Normal) [41] ZHANG H, LV Y, GUO J. New development direction of underground logistics from the perspective of public transport: a systematic review based on scientometrics[J]. Sustainability, 2022, 14(6): 3179.

[64] (Normal) [42] ZHENG C, GU Y, SHEN J, et al. Urban logistics delivery route planning based on a single metro line[J]. IEEE Access, 2021, 9: 50819-50830.

[65] (Normal) [43] ZHANG H, ZHANG J, ZHENG C, et al. Node location of bi-level urban metro-based ground-underground logistics distribution[J]. Multimodal Transportation, 2024, 3(1): 100119.

[66] (Normal) [44] BAI Z, WANG H, YANG K. Optimizing metro-based logistics hub locations for sustainable urban freight distribution[J]. Sustainability, 2025, 17(10): 4735.

[67] (Normal) [45] TIAN J, XUE S, SHIWAKOTI N. Toward sustainable urban freight: a two-stage station-network design framework for metro-based underground logistics system[J]. Sustainable Cities and Society, 2026, 120: 107436.

[68] (Normal) [46] YANG X, YUAN M, GAN M, et al. New options for metro-based underground logistics system layout strategies driven by supply-demand relationships: induced, derived, or balanced[J]. Case Studies on Transport Policy, 2025, 22: 101628.

[69] (Normal) [47] HU Q M, GUO S. Design of incomplete hub location-routing networks in an intra-city metro logistics system[J]. European Journal of Industrial Engineering, 2025, 20(4): 590-652.

[70] (Normal) [48] MA M, ZHANG F, LIU W, et al. A game theoretical analysis of metro-integrated city logistics systems[J]. Transportation Research Part B: Methodological, 2022, 156: 14-27.

[71] (Normal) [49] YE Y, GUO J, YAN L. A metro freight plan for mixed passenger and freight transportation[J]. Journal of Advanced Transportation, 2022, 2022: 6602387.

[72] (Normal) [50] YANG K, CHEN Y, HU W, et al. Wasserstein distributionally robust optimization for train operation and freight assignment in a metro-based underground logistics system[J]. Computers & Industrial Engineering, 2024, 196: 110228.

[73] (Normal) [51] LI S, ZHU X, SHANG P. Joint optimization of train services and freight delivery in a metro-based underground logistics system[J]. Transportation Research Part B: Methodological, 2026, 199: 103377.

[74] (Normal) [52] HU W, REN R, YANG J. Unequal-area facility layout optimization for a freight-passenger integrated metro station considering logistics functional partitioning and device allotment: a bi-level programming method[J]. Computers & Industrial Engineering, 2025, 204: 111189.

[75] (Normal) [53] DU X, YANG L, WANG X, et al. Pricing and capacity optimization for underground logistics[J]. Transportation Research Part E: Logistics and Transportation Review, 2025, 203: 104386.

[76] (Normal) [54] LI J, WANG Y, LIU H. The optimization of split demand route planning under metro-truck collaborative distribution[J]. Urban Rail Transit, 2024, 10(2): 144-159.

[77] (Normal) [55] HOU L, HU W, CHEN Y. Measuring effectiveness of metro-based underground logistics system in sustaining city logistics performance during public health emergencies: a case study of Shanghai[J]. Transportation Research Record, 2024.

[78] (Normal) [56] SUN K, GU Y, WAN FEI K. Medical supplies delivery route optimization under public health emergencies incorporating metro-based logistics system[J]. Transportation Research Record, 2024.

[79] (Normal) [57] LI J, DONG J, REN R. Modeling resilience of metro-based urban underground logistics system based on multi-layer interdependent network[J]. Sustainability, 2024, 16(22): 9892.

[80] (Normal) [58] DERSE O, VAN WOENSEL T. Integrated people and freight transportation: a literature review[J]. Future Transportation, 2024, 4(4): 1254-1309.

[81] (Normal) [59] HU H, WANG J. Research on the design and sustainable evaluation of metro-based underground logistics systems[J]. IEEE Access, 2023, 11: 67600-67612.

[82] (Normal) [60] SUN X, LIU S, WEI L, et al. Tripartite evolutionary game and simulation analysis of stakeholder strategy implementation in metro-based freight systems considering low-carbon benefits[J]. Systems, 2026, 14(4): 437.