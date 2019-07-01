import pandas as pd
import numpy as np
from ParseData import *
from sklearn.cluster import KMeans

clist = ['XQ02988', 'XQ00082', 'XQ01247', 'XQ00998', 'XQ02507', 'XQ01527', 'XQ02716', 'XQ03389', 'XQ03784', 'XQ03569', 'XQ03857', 'XQ02218', 'XQ01239', 'XQ01280', 'XQ03358', 'XQ01689', 'XQ02123', 'XQ01912', 'XQ04015', 'XQ03324', 'XQ01969', 'XQ00742', 'XQ02636', 'XQ00453', 'XQ01318', 'XQ02468', 'XQ01089', 'XQ02931', 'XQ01254', 'XQ01710', 'XQ02162', 'XQ01391', 'XQ00042', 'XQ01147', 'XQ00552', 'XQ02790', 'XQ03317', 'XQ02898', 'XQ01158', 'XQ03374', 'XQ00498', 'XQ02758', 'XQ01884', 'XQ02977', 'XQ02274', 'XQ00221', 'XQ00334', 'XQ01917', 'XQ01934', 'XQ03401', 'XQ02147', 'XQ00368', 'XQ01165', 'XQ01366', 'XQ01118', 'XQ00040', 'XQ02157', 'XQ01539', 'XQ04007', 'XQ02866', 'XQ01406', 'XQ01347', 'XQ03626', 'XQ04087', 'XQ01183', 'XQ03861', 'XQ00157', 'XQ02820', 'XQ00158', 'XQ00536', 'XQ00735', 'XQ00869', 'XQ03150', 'XQ02722', 'XQ03404', 'XQ01713', 'XQ02989', 'XQ03750', 'XQ02869', 'XQ03944', 'XQ03863', 'XQ00737', 'XQ00537', 'XQ03690', 'XQ03565', 'XQ02943', 'XQ02774', 'XQ00884', 'XQ04091', 'XQ01898', 'XQ02108', 'XQ02163', 'XQ01106', 'XQ02792', 'XQ00464', 'XQ03021', 'XQ01311', 'XQ03887', 'XQ00052', 'XQ04076', 'XQ00849', 'XQ02899', 'XQ02950', 'XQ04053', 'XQ01033', 'XQ03070', 'XQ03322', 'XQ00191', 'XQ01889', 'XQ03711', 'XQ04178', 'XQ03852', 'XQ01474', 'XQ01138', 'XQ02176', 'XQ02573', 'XQ01179', 'XQ04196', 'XQ04168', 'XQ03856', 'XQ00070', 'XQ00109', 'XQ03843', 'XQ02003', 'XQ02704', 'XQ01925', 'XQ02965', 'XQ03630', 'XQ03814', 'XQ02719', 'XQ01378', 'XQ01127', 'XQ03691', 'XQ01429', 'XQ01479', 'XQ03472', 'XQ03313', 'XQ01897', 'XQ01528', 'XQ02720', 'XQ02759', 'XQ02682', 'XQ00448', 'XQ02647', 'XQ02872', 'XQ00499', 'XQ02779', 'XQ02149', 'XQ03046', 'XQ00551', 'XQ00719', 'XQ03349', 'XQ02148', 'XQ02959', 'XQ00056', 'XQ02786', 'XQ02875', 'XQ01423', 'XQ02857', 'XQ02466', 'XQ01103', 'XQ01352', 'XQ02581', 'XQ01822', 'XQ02828', 'XQ02508', 'XQ03515', 'XQ01971', 'XQ02143', 'XQ02254', 'XQ01170', 'XQ02600', 'XQ01691', 'XQ00174', 'XQ02467', 'XQ02524', 'XQ00883', 'XQ02750', 'XQ02985', 'XQ00081', 'XQ04096', 'XQ02153', 'XQ04208', 'XQ04183', 'XQ01705', 'XQ00116', 'XQ01284', 'XQ03211', 'XQ02224', 'XQ04152', 'XQ03030', 'XQ04179', 'XQ02954', 'XQ04077', 'XQ00141', 'XQ00747', 'XQ02202', 'XQ00678', 'XQ00189', 'XQ00455', 'XQ02867', 'XQ02757', 'XQ04086', 'XQ02919', 'XQ03363', 'XQ01218', 'XQ00145', 'XQ03287', 'XQ02855', 'XQ03755', 'XQ02030', 'XQ00531', 'XQ02020', 'XQ00454', 'XQ03055', 'XQ01303', 'XQ04013', 'XQ04093', 'XQ02120', 'XQ03994', 'XQ00940', 'XQ00061', 'XQ03054', 'XQ04067', 'XQ03384', 'XQ04063', 'XQ03408', 'XQ01146', 'XQ02764', 'XQ02465', 'XQ03227', 'XQ02802', 'XQ02567', 'XQ02164', 'XQ01242', 'XQ03425', 'XQ04006', 'XQ03916', 'XQ03734', 'XQ03223', 'XQ02670', 'XQ01046', 'XQ01379', 'XQ02008', 'XQ04174', 'XQ01268', 'XQ04064', 'XQ03650', 'XQ00790', 'XQ00105', 'XQ00333', 'XQ03532', 'XQ01978', 'XQ04180', 'XQ02347', 'XQ00555', 'XQ02815', 'XQ03151', 'XQ00867', 'XQ01789', 'XQ02114', 'XQ00934', 'XQ02880', 'XQ00073', 'XQ01682', 'XQ01881', 'XQ00412', 'XQ02144', 'XQ01824', 'XQ01098', 'XQ04088', 'XQ03388', 'XQ01007', 'XQ02849', 'XQ00358', 'XQ01238', 'XQ02886', 'XQ02752', 'XQ02742', 'XQ00078', 'XQ01109', 'XQ00741', 'XQ01172', 'XQ02910', 'XQ02843', 'XQ04073', 'XQ00362', 'XQ02255', 'XQ00195', 'XQ04047', 'XQ02770', 'XQ00987', 'XQ04099', 'XQ02918', 'XQ03212', 'XQ01976', 'XQ02840', 'XQ00112', 'XQ02966', 'XQ04008', 'XQ03587', 'XQ00097', 'XQ04120', 'XQ03335', 'XQ03326', 'XQ00176', 'XQ01134', 'XQ00989', 'XQ02572', 'XQ02055', 'XQ01237', 'XQ02599', 'XQ02588', 'XQ01816', 'XQ03649', 'XQ03247', 'XQ04160', 'XQ01174', 'XQ03153', 'XQ04203', 'XQ01153', 'XQ00170', 'XQ02597', 'XQ00465', 'XQ02302', 'XQ01800', 'XQ03314', 'XQ01000', 'XQ01863', 'XQ01485', 'XQ04060', 'XQ03350', 'XQ01937', 'XQ03424', 'XQ03325', 'XQ02829', 'XQ04040', 'XQ01325', 'XQ01001', 'XQ03473', 'XQ03340', 'XQ02142', 'XQ00541', 'XQ02761', 'XQ02132', 'XQ03902', 'XQ01882', 'XQ04113', 'XQ01175', 'XQ02851', 'XQ02175', 'XQ02295', 'XQ01417', 'XQ04075', 'XQ04095', 'XQ03992', 'XQ02651', 'XQ02593', 'XQ03342', 'XQ03821', 'XQ01895', 'XQ02607', 'XQ01973', 'XQ04078', 'XQ04055', 'XQ01718', 'XQ04107', 'XQ03615', 'XQ04070', 'XQ04089', 'XQ03028', 'XQ03074', 'XQ00508', 'XQ02139', 'XQ00866', 'XQ00738', 'XQ03012', 'XQ02078', 'XQ03399', 'XQ02514', 'XQ00846', 'XQ02975', 'XQ02257', 'XQ01141', 'XQ00554', 'XQ02796', 'XQ01094', 'XQ01947', 'XQ03535', 'XQ01144', 'XQ02789', 'XQ01741', 'XQ03283', 'XQ01641', 'XQ00500', 'XQ03191', 'XQ02911', 'XQ00032', 'XQ01924', 'XQ02974', 'XQ00492', 'XQ04085', 'XQ01353', 'XQ01160', 'XQ02920', 'XQ02141', 'XQ03860', 'XQ02971', 'XQ04025', 'XQ01403', 'XQ01187', 'XQ00356', 'XQ02102', 'XQ00526', 'XQ02015', 'XQ02854', 'XQ03347', 'XQ02166', 'XQ04018', 'XQ01181', 'XQ02765', 'XQ00004', 'XQ03064', 'XQ02946', 'XQ01510', 'XQ02640', 'XQ03188', 'XQ01640', 'XQ04014', 'XQ03660', 'XQ01157', 'XQ03997', 'XQ04050', 'XQ04103', 'XQ02878', 'XQ04105', 'XQ00826', 'XQ03980', 'XQ03581', 'XQ02167', 'XQ01143', 'XQ01248', 'XQ01151', 'XQ01014', 'XQ00161', 'XQ01295', 'XQ01097', 'XQ02177', 'XQ00179', 'XQ02797', 'XQ01148', 'XQ01312', 'XQ00562', 'XQ00340', 'XQ03154', 'XQ01994', 'XQ01965', 'XQ03736', 'XQ03976', 'XQ01684', 'XQ00538', 'XQ02219', 'XQ03040', 'XQ03575', 'XQ03770', 'XQ01235', 'XQ03812', 'XQ00325', 'XQ02155', 'XQ01167', 'XQ03746', 'XQ00142', 'XQ04100', 'XQ01704', 'XQ01101', 'XQ03372', 'XQ01622', 'XQ04169', 'XQ03862', 'XQ01589', 'XQ04010', 'XQ03362', 'XQ01117', 'XQ04034', 'XQ03042', 'XQ03323', 'XQ00067', 'XQ02860', 'XQ03704', 'XQ03915', 'XQ03614', 'XQ04083', 'XQ02220', 'XQ00512', 'XQ00114', 'XQ03373', 'XQ00113', 'XQ04084', 'XQ03824', 'XQ04002', 'XQ04155', 'XQ03759', 'XQ02156', 'XQ02743', 'XQ02516', 'XQ03241', 'XQ00802', 'XQ01806', 'XQ02130', 'XQ01941', 'XQ02795', 'XQ03811', 'XQ01476', 'XQ00360', 'XQ01156', 'XQ03998', 'XQ01302', 'XQ03383', 'XQ04001', 'XQ01792', 'XQ00877', 'XQ01939', 'XQ03307', 'XQ02052', 'XQ02201', 'XQ02882', 'XQ01833', 'XQ00392','XQ03918', 'XQ02748', 'XQ00799', 'XQ03006', 'XQ03871', 'XQ01782', 'XQ01123', 'XQ04071', 'XQ03817', 'XQ00377', 'XQ03651', 'XQ01940', 'XQ02865', 'XQ00493', 'XQ03688', 'XQ03041', 'XQ00359', 'XQ00079', 'XQ00376', 'XQ02844', 'XQ02121', 'XQ01136', 'XQ02124', 'XQ00789', 'XQ01085', 'XQ01180', 'XQ00185', 'XQ02237', 'XQ03019', 'XQ02960', 'XQ02830', 'XQ00540', 'XQ04092', 'XQ03884', 'XQ03449', 'XQ04181', 'XQ02113', 'XQ00426', 'XQ02039', 'XQ02110', 'XQ04193', 'XQ02197', 'XQ01241', 'XQ03034', 'XQ03892', 'XQ01206', 'XQ03589', 'XQ02818', 'XQ02883', 'XQ03943', 'XQ01364', 'XQ02211', 'XQ02863', 'XQ03332', 'XQ00457', 'XQ03079', 'XQ00446', 'XQ03360', 'XQ01696', 'XQ02819', 'XQ03371', 'XQ01873', 'XQ01003', 'XQ00115', 'XQ00743', 'XQ01784', 'XQ00339', 'XQ04231', 'XQ02299', 'XQ02601', 'XQ03017', 'XQ04036', 'XQ02776', 'XQ03749', 'XQ02298', 'XQ00865', 'XQ00406', 'XQ02858', 'XQ02831', 'XQ00986', 'XQ00507', 'XQ01910', 'XQ04097', 'XQ02022', 'XQ03758', 'XQ01714', 'XQ01102', 'XQ04101', 'XQ02970', 'XQ03357', 'XQ03818', 'XQ01872', 'XQ03348', 'XQ00730', 'XQ01938', 'XQ02654', 'XQ02885', 'XQ01690', 'XQ01137', 'XQ03761', 'XQ00847', 'XQ02604', 'XQ03354', 'XQ01979', 'XQ03643', 'XQ00053', 'XQ00925', 'XQ03845', 'XQ01432', 'XQ02686', 'XQ03366', 'XQ00177', 'XQ01791', 'XQ04011', 'XQ00171', 'XQ03886', 'XQ02901', 'XQ02140', 'XQ01377', 'XQ00162', 'XQ03245', 'XQ02045', 'XQ02239', 'XQ04017', 'XQ03187', 'XQ03667', 'XQ02982', 'XQ00155', 'XQ00881', 'XQ00893', 'XQ02811', 'XQ03020', 'XQ03919', 'XQ01171', 'XQ01052', 'XQ03382', 'XQ00143', 'XQ03516', 'XQ03791', 'XQ01263', 'XQ01088', 'XQ00466', 'XQ02111', 'XQ03456', 'XQ01707', 'XQ01455', 'XQ02075', 'XQ00190', 'XQ02633', 'XQ03376', 'XQ04059', 'XQ03873', 'XQ02628', 'XQ03948', 'XQ03459', 'XQ01132', 'XQ01932', 'XQ01020', 'XQ03331', 'XQ02842', 'XQ02979', 'XQ02738', 'XQ00528', 'XQ01105', 'XQ02179', 'XQ02209', 'XQ03011', 'XQ03375', 'XQ02862', 'XQ03337', 'XQ02032', 'XQ01231', 'XQ04186', 'XQ03370', 'XQ02603', 'XQ03394', 'XQ01887', 'XQ03333', 'XQ02766', 'XQ00495', 'XQ00539', 'XQ03900', 'XQ03138', 'XQ03329', 'XQ02118', 'XQ02721', 'XQ04023', 'XQ02848', 'XQ02552', 'XQ01091', 'XQ00045', 'XQ04206', 'XQ02826', 'XQ01129', 'XQ00556', 'XQ02542', 'XQ01135', 'XQ01783', 'XQ03004', 'XQ00971', 'XQ02131', 'XQ00335', 'XQ02154', 'XQ02206', 'XQ02210', 'XQ03195', 'XQ02837', 'XQ02550', 'XQ00022', 'XQ02247', 'XQ03961', 'XQ02700', 'XQ02002', 'XQ00367', 'XQ02915', 'XQ03645', 'XQ03639', 'XQ02540', 'XQ02236', 'XQ02887', 'XQ03609', 'XQ00709', 'XQ00775', 'XQ00514', 'XQ01972', 'XQ01013', 'XQ03789', 'XQ00164', 'XQ00553', 'XQ00966', 'XQ04225', 'XQ00422', 'XQ04039', 'XQ03400', 'XQ04020', 'XQ03642', 'XQ02109', 'XQ01229', 'XQ02744', 'XQ00739', 'XQ02145', 'XQ03066', 'XQ03018', 'XQ00997', 'XQ04185', 'XQ00949', 'XQ00999', 'XQ03634', 'XQ00180', 'XQ02972', 'XQ01246', 'XQ02879', 'XQ01086', 'XQ00068', 'XQ03044', 'XQ00736', 'XQ04202', 'XQ01434', 'XQ02598', 'XQ02214', 'XQ01807', 'XQ03352', 'XQ02922', 'XQ01915', 'XQ03826', 'XQ03068', 'XQ01298', 'XQ00106', 'XQ01828', 'XQ04065', 'XQ03065', 'XQ02914', 'XQ03359', 'XQ01099', 'XQ01790', 'XQ01354', 'XQ00963', 'XQ03378', 'XQ02112', 'XQ03972', 'XQ03367', 'XQ02291', 'XQ04000', 'XQ01244', 'XQ02158', 'XQ04090', 'XQ00824', 'XQ01398', 'XQ00149', 'XQ02900', 'XQ01236', 'XQ00355', 'XQ01797', 'XQ00981', 'XQ02190', 'XQ02119', 'XQ01168', 'XQ00041', 'XQ02981', 'XQ02192', 'XQ01692', 'XQ01217', 'XQ03078', 'XQ01042', 'XQ02122', 'XQ03457', 'XQ03686', 'XQ00001', 'XQ02868', 'XQ02053', 'XQ01155', 'XQ00046', 'XQ00095', 'XQ04031', 'XQ01119', 'XQ00077', 'XQ01911', 'XQ03891', 'XQ02803', 'XQ00885', 'XQ03417', 'XQ03564', 'XQ02968', 'XQ01286', 'XQ00985', 'XQ02447', 'XQ03062', 'XQ00014', 'XQ03659', 'XQ00744', 'XQ00101', 'XQ01152', 'XQ00361', 'XQ03507', 'XQ03067', 'XQ00425', 'XQ00852', 'XQ01695', 'XQ01980', 'XQ04104', 'XQ03813', 'XQ00520', 'XQ01169', 'XQ00154', 'XQ00524', 'XQ03573', 'XQ04003', 'XQ00523', 'XQ03392', 'XQ01802', 'XQ04098', 'XQ01240', 'XQ04068', 'XQ02100', 'XQ03790', 'XQ01327', 'XQ02723', 'XQ03653', 'XQ04198', 'XQ00527', 'XQ01759', 'XQ00232', 'XQ00401', 'XQ01798', 'XQ00188', 'XQ01329', 'XQ01550', 'XQ02944', 'XQ00048', 'XQ03003', 'XQ03259', 'XQ03751', 'XQ02509', 'XQ03361', 'XQ02270', 'XQ02782', 'XQ01133', 'XQ03775', 'XQ02161', 'XQ00182', 'XQ00819', 'XQ02558', 'XQ02655', 'XQ03355', 'XQ03338', 'XQ03147', 'XQ04072', 'XQ02594', 'XQ03623', 'XQ04022', 'XQ04024', 'XQ00369', 'XQ01245', 'XQ03578', 'XQ02128', 'XQ03060', 'XQ01149', 'XQ03753', 'XQ01513', 'XQ01444', 'XQ01951', 'XQ02637', 'XQ02457', 'XQ01131', 'XQ00543', 'XQ01162', 'XQ04079', 'XQ00946', 'XQ02278', 'XQ02955', 'XQ01044', 'XQ03351', 'XQ04081', 'XQ01261', 'XQ01374', 'XQ01439', 'XQ01516', 'XQ03418', 'XQ01435', 'XQ02505', 'XQ00055', 'XQ02553', 'XQ00178', 'XQ03865', 'XQ01142', 'XQ03010', 'XQ03835', 'XQ03774', 'XQ03224', 'XQ02659', 'XQ01228', 'XQ00151', 'XQ03967', 'XQ04094', 'XQ02260', 'XQ00181', 'XQ03769', 'XQ00606', 'XQ04009', 'XQ04049', 'XQ03765', 'XQ04074', 'XQ03537', 'XQ02671', 'XQ04052', 'XQ00777', 'XQ02993', 'XQ03426', 'XQ03872', 'XQ04102', 'XQ00088', 'XQ00956', 'XQ04046', 'XQ03026', 'XQ00044', 'XQ03008', 'XQ03494', 'XQ02732', 'XQ01635', 'XQ00047', 'XQ00194', 'XQ00534', 'XQ02769', 'XQ01477', 'XQ02967', 'XQ02160', 'XQ02595', 'XQ01385', 'XQ02976', 'XQ01390', 'XQ00031', 'XQ01437', 'XQ03756', 'XQ03783', 'XQ03533', 'XQ00163', 'XQ02614', 'XQ02519', 'XQ02990', 'XQ02859', 'XQ03945', 'XQ01234', 'XQ01864', 'XQ00517', 'XQ03069', 'XQ00409', 'XQ00772', 'XQ02798', 'XQ03035', 'XQ01154', 'XQ02841', 'XQ02702', 'XQ00467', 'XQ02138', 'XQ03489', 'XQ03005', 'XQ01430', 'XQ02708', 'XQ01173', 'XQ01380', 'XQ03258', 'XQ00933', 'XQ02775', 'XQ03903', 'XQ01342', 'XQ00496', 'XQ03619', 'XQ03754', 'XQ03625', 'XQ03341', 'XQ00146', 'XQ01826', 'XQ03430', 'XQ02125', 'XQ00160', 'XQ01779', 'XQ02699', 'XQ03973', 'XQ03391', 'XQ02864', 'XQ00341', 'XQ03825', 'XQ02001', 'XQ01974', 'XQ03381', 'XQ04184', 'XQ02817', 'XQ04151', 'XQ04030']


# 特征工程
def feature(df):
    """
    ######之前做的特征######
    df['高级学校占比'] = (df['interSchoolNum'] + df['privateSchoolNum']) / (df['schoolNum'] + df['interSchoolNum'] + df['privateSchoolNum'])
    df['该板块网页重复查看次数'] = df['lookNum'] + min_max_scaler(df['pv'] / df['uv'])*3

    df['卧室占比'] = df['room_num'] / df['房间总数']
    df['房间总数'] = df['room_num'] + df['hall_num'] + df['bathroom_num']
    df['建筑年份'] = df['buildYear'].apply(lambda x: 2018-x)

    df['新房售出比例'] = df['tradeNewNum'] / (df['tradeNewNum'] + df['remainNewNum'])
    df['新房交易面积占比'] = df['totalNewTradeArea'] / (df['totalTradeArea']+df['totalNewTradeArea'])
    df['新房交易金额占比'] = df['totalNewTradeMoney'] / (df['totalTradeMoney']+df['totalNewTradeMoney'])
    df['新房二手房交易平均价格差'] = (df['tradeNewMeanPrice'] - df['tradeMeanPrice']) / (df['tradeNewMeanPrice'] + df['tradeMeanPrice'])

    df['人均公交车站数'] = df['busStationNum'] / (df['totalWorkers']) * 10000
    df['人均房屋面积'] = (df['totalTradeArea'] + df['totalNewTradeArea']) / (df['newWorkersProportion'] + df['totalWorkers'])
    #df['人均房屋面积'] = (df['totalTradeArea'] + df['totalNewTradeArea']) / (df['newWorkersProportion'])
    df['人均超市'] = df['consumePlaceRate'] / (df['residentPopulation'] + df['totalWorkers']) * 10000
    df['人均银行和健身房'] = (df['bankNum'] + df['gymNum']) / df['totalWorkers'] * 10000
    df['人均健身房和公园'] = (df['parkNum'] + df['gymNum']) / (df['totalWorkers']) * 10000

    df['办公群体占比'] = df['totalWorkers'] / (df['totalWorkers'] + df['residentPopulation'])
    df['新流入人员占比'] = df['newWorkers'] / df['totalWorkers']
    """

    df['是否高档小区'] = 0
    df['是否高档小区'][df['communityName'].apply(lambda x: x in clist)] = 1

    tmp = df.copy()
    tmp['trueArea'] = None
    i = tmp['area'] <= 20
    tmp['trueArea'][i] = tmp['area'][i] * (tmp['room_num'][i] + tmp['hall_num'][i] + tmp['bathroom_num'][i]/3)
    tmp['trueArea'][tmp['area'] > 20] = tmp['area'][tmp['area'] > 20]

    df['交易月份'] = df['tradeTime'].apply(lambda x: int(x.split('/')[1]))

    df['小区交易数/板块交易数'] = df['communityName_count'] / df['plate_count']
    df.drop(['communityName_count', 'plate_count', 'region_count'], axis=1, inplace=True)
    df['交易数除以看房数'] = (df['tradeNewNum']+df['tradeSecNum']) / df['uv']

    df['房间面积'] = tmp['trueArea'] / (df['room_num'] + df['hall_num'] + df['bathroom_num'])
    df['房间面积'] = df['房间面积'].astype('float')

    df.drop('tradeTime', axis=1, inplace=True)
    df.drop('communityName', axis=1, inplace=True)

    df['buildYear'] = df['buildYear'].astype('category')

    return df, ['rentType', 'houseToward', 'houseDecoration', 'plate', 'region', 'houseFloor', 'buildYear']


# 获取上个月二手房每平米交易均值
def getLastMonthTradeMeanPrice(df_train, df_test):
    """
    获取上个月二手房每平米交易均值
    """
    train = df_train.copy()
    train.drop('tradeMoney',axis=1,inplace=True)
    test = df_test.copy()
    lastMonthTradeMeanPrice = pd.concat([train,test],ignore_index=True).groupby(['plate', '交易月份']).first()
    lastMonthTradeMeanPrice = pd.DataFrame(lastMonthTradeMeanPrice['tradeMeanPrice'])
    lastMonthTradeMeanPrice.reset_index(inplace=True)
    lastMonthTradeMeanPrice.rename(columns={'tradeMeanPrice': 'lastMonthTradeMeanPrice'}, inplace=True)
    lastMonthTradeMeanPrice['交易月份'] = lastMonthTradeMeanPrice['交易月份'].apply(lambda x: x - 1 if x - 1 > 0 else 12)
    lastMonthTradeMeanPrice['lastMonthTradeMeanPrice'].fillna(0, inplace=True)

    df_train = pd.merge(df_train, lastMonthTradeMeanPrice, on=['plate', '交易月份'])
    df_test = pd.merge(df_test, lastMonthTradeMeanPrice, how='left', on=['plate', '交易月份'])

    df_train['lastMonthTradeMeanPrice'].fillna(0, inplace=True)
    df_test['lastMonthTradeMeanPrice'].fillna(0, inplace=True)
    return df_train, df_test


# 将col中的特征转换为该类别出现的总次数
def encode_feature(df_train, df_test, col, cat_feats):
    """
    将col中的特征转化为该类别出现的总次数
    """
    cv = df_train[col].value_counts()
    df_train[col] = df_train[col].map(cv)
    df_test[col] = df_test[col].map(cv)
    df_test[col].fillna(0, inplace=True)
    cat_feats.remove(col)
    return


#添加feat中特征各种类别的出现次数作为新特征
def getNumOfFeat(df_train, df_test, feat):
    tmp = pd.concat([df_train, df_test])
    cv = tmp[feat].value_counts()
    df_train[feat+"_count"] = df_train[feat].map(cv)
    df_test[feat+"_count"] = df_test[feat].map(cv)
    df_test[feat+"_count"].fillna(0, inplace=True)


# 通过聚类创造新特征
def classifyByKMeans(df_train, df_test, features, categorical_feats, groupFeats, name, n_clusters=3):
    """
    利用KMeans无监督聚类
    :param groupFeats: 根据这些特征进行聚类
    :param name: 生成的特征名称
    :param n_clusters: 聚类数量
    """
    feats = groupFeats
    train = df_train[feats].copy()
    test = df_test[feats].copy()

    tmp = []
    for col in train.columns:
        if col in categorical_feats:
            tmp.append(col)
    categorical_feats = tmp.copy()

    train = onehotCat(train, categorical_feats)
    test = onehotCat(test, categorical_feats)
    train, test = fillColumns(train, test)

    model = KMeans(n_clusters=n_clusters)
    model.fit(train)
    predicted_label = model.predict(train)

    df_train[name] = predicted_label
    df_train[name] = df_train[name].astype('category')

    predicted_label = model.predict(test)

    df_test[name] = predicted_label
    df_test[name] = df_test[name].astype('category')

    features.append(name)
    categorical_feats.append(name)
    return


# 计算板块平均修建年份
def getMeanFloor(df_train, df_test):
    """
    计算板块平均修建年份
    """
    train = df_train.copy()
    train.drop('tradeMoney', axis=1, inplace=True)
    df = pd.concat([train, df_test])
    data = df.groupby(['plate']).mean().reset_index()[['plate', 'totalFloor']]
    data['totalFloor'] = data['totalFloor'].apply(round)
    data.rename(columns={'totalFloor': '板块平均楼高'}, inplace=True)

    df_train = pd.merge(df_train, data, on=['plate'], how='left')
    df_test = pd.merge(df_test, data, on=['plate'], how='left')
    df_train['plate'] = df_train['plate'].astype('category')
    df_test['plate'] = df_test['plate'].astype('category')

    return df_train, df_test


# 计算当月二手房成交比例
def getSecondHandRatio(df_train, df_test):
    """
    计算当月二手房成交比例
    """
    tmp = df_train.copy().drop('tradeMoney',axis=1)
    df = pd.concat([tmp,df_test])

    data = df.groupby(['plate']).first().reset_index()
    data['二手房成交比例'] = data['tradeSecNum'] / (data['tradeSecNum'] + data['tradeNewNum'] + 1)
    data = data[['plate','二手房成交比例']]

    df_train = pd.merge(df_train, data, on=['plate'], how='left')
    df_test = pd.merge(df_test, data, on=['plate'], how='left')
    df_train['plate'] = df_train['plate'].astype('category')
    df_test['plate'] = df_test['plate'].astype('category')

    return df_train, df_test


# 比例特征
def getTradeMoneyRatio(df_train, df_test):
    tmp1 = df_train[['communityName','tradeMoney']].copy()
    data1 = tmp1.groupby(['communityName']).sum()
    data1.reset_index(inplace=True)
    data1.rename(columns={'tradeMoney':'小区交易总额'}, inplace=True)
    df_train = pd.merge(df_train, data1[['communityName','小区交易总额']], on=['communityName'], how='left')
    df_test = pd.merge(df_test, data1[['communityName','小区交易总额']], on=['communityName'], how='left')

    tmp2 = df_train[['plate','tradeMoney']].copy()
    data2 = tmp2.groupby(['plate']).sum()
    data2.reset_index(inplace=True)
    data2.rename(columns={'tradeMoney':'板块交易总额'}, inplace=True)
    df_train = pd.merge(df_train, data2[['plate','板块交易总额']], on=['plate'], how='left')
    df_test = pd.merge(df_test, data2[['plate','板块交易总额']], on=['plate'], how='left')

    df_train['小区交易额板块占比'] = df_train['小区交易总额'] / df_train['板块交易总额']
    df_test['小区交易额板块占比'] = df_test['小区交易总额'] / df_test['板块交易总额']

    df_train.drop(['小区交易总额','板块交易总额'],axis=1,inplace=True)
    df_test.drop(['小区交易总额','板块交易总额'],axis=1,inplace=True)

    df_train['plate'] = df_train['plate'].astype('category')
    df_test['plate'] = df_test['plate'].astype('category')

    return df_train, df_test


# 板块均房间
def getPlateMeanRoom(df_train, df_test):
    tmp = pd.concat([df_train, df_test])[['plate','room_num','bathroom_num','hall_num']]
    data = tmp.groupby(['plate']).mean()
    data.reset_index(inplace=True)
    data.rename(columns={'room_num':'室平均','bathroom_num':"卫平均",'hall_num':"厅平均"},inplace=True)

    df_train = pd.merge(df_train, data, on=['plate'], how='left')
    df_test = pd.merge(df_test, data, on=['plate'], how='left')

    df_train['plate'] = df_train['plate'].astype('category')
    df_test['plate'] = df_test['plate'].astype('category')

    return df_train, df_test


# 将columns中的特征独热编码
def onehotCat(df, columns):
    """
    将columns中的特征独热编码
    """
    for col in columns:
        pf = pd.get_dummies(df[col])
        pf = pf.astype('float')
        df = pd.concat([df, pf], axis=1)
        df.drop(col, axis=1, inplace=True)

    return df


# 独热编码特征补全
def fillColumns(df_train, df_test):
    # 将类别特征独热编码后，将测试集中缺少的列加入到测试集
    train_col = df_train.columns
    test_col = df_test.columns
    for col in train_col:
        if (col not in test_col) and col != 'tradeMoney':
            df_test[col] = 0
    for col in test_col:
        if (col not in train_col) and col != 'tradeMoney':
            df_train[col] = 0
    return df_train, df_test


############ 其他特征工程 ###############
def getRecentTradeMoney(df_train, df_test):
    """
    获取该板块近10天的交易均值
    """
    train = pd.read_csv('src/train_data.csv')

    train['parsedTime'] = pd.to_datetime(train['tradeTime'],format="%Y/%m/%d")
    df_train['parsedTime'] = pd.to_datetime(df_train['tradeTime'],format="%Y/%m/%d")
    df_test['parsedTime'] = pd.to_datetime(df_test['tradeTime'],format="%Y/%m/%d")

    data1 = train.groupby(['plate','parsedTime']).mean()
    data2 = data1.reset_index()
    data2['近10天平均交易金额'] = 0

    def getMeanData(x):
        d1 = x['parsedTime'] - np.timedelta64(5, 'D')
        d2 = x['parsedTime'] - np.timedelta64(4, 'D')
        d3 = x['parsedTime'] - np.timedelta64(3, 'D')
        d4 = x['parsedTime'] - np.timedelta64(2, 'D')
        d5 = x['parsedTime'] - np.timedelta64(1, 'D')
        d6 = x['parsedTime'] - np.timedelta64(0, 'D')
        d7 = x['parsedTime'] + np.timedelta64(1, 'D')
        d8 = x['parsedTime'] + np.timedelta64(2, 'D')
        d9 = x['parsedTime'] + np.timedelta64(3, 'D')
        d10 = x['parsedTime'] + np.timedelta64(4, 'D')
        d11 = x['parsedTime'] + np.timedelta64(5, 'D')
        l = [d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11]
        res = data1.loc[x['plate']].loc[l]['tradeMoney'].mean()
        return res

    data2['近10天平均交易金额'] = data2.apply(getMeanData,axis=1)
    df_train = pd.merge(df_train, data2[['plate','parsedTime','近10天平均交易金额']], on=['plate','parsedTime'],how='left')
    df_test = pd.merge(df_test, data2[['plate','parsedTime','近10天平均交易金额']], on=['plate','parsedTime'],how='left')

    df_train['plate'] = df_train['plate'].astype('category')
    df_test['plate'] = df_test['plate'].astype('category')
    df_train.drop('parsedTime',axis=1,inplace=True)
    df_test.drop('parsedTime',axis=1,inplace=True)
    return df_train, df_test


def getPlateBuildYearMean(train, test):
    """
    计算同一板块中相同修建年份的房屋的平均交易金额
    """
    tr = pd.read_csv('src/train_data.csv')
    tr = parseData(tr)

    d = tr.groupby(['plate', 'buildYear']).mean()
    d.reset_index(inplace=True)
    d.dropna(inplace=True)
    d = d[['plate', 'buildYear', 'tradeMoney']]
    d.rename(columns={"tradeMoney": "板块相同年份平均交易金额"}, inplace=True)

    train = pd.merge(train, d, on=['plate', 'buildYear'], how='left')
    train['plate'] = train['plate'].astype('category')
    train['buildYear'] = train['buildYear'].astype('category')
    train['板块相同年份平均交易金额'] = train['板块相同年份平均交易金额'].astype('int')

    test = pd.merge(test, d, on=['plate', 'buildYear'], how='left')
    test['plate'] = test['plate'].astype('category')
    test['buildYear'] = test['buildYear'].astype('category')
    test['板块相同年份平均交易金额'] = test['板块相同年份平均交易金额'].astype('int')

    return train, test


def getSumOfPlateData(train, test, feat):
    feat_data = train.groupby([feat, '交易月份', 'plate']).first() \
        .groupby([feat, '交易月份']).sum()
    feats = ['totalTradeMoney', 'totalTradeArea', 'tradeMeanPrice',
             'tradeSecNum', 'totalNewTradeMoney', 'totalNewTradeArea',
             'tradeNewMeanPrice', 'tradeNewNum', 'remainNewNum']
    feat_data = feat_data[feats]
    nameMap = {}
    for f in feats:
        nameMap[f] = feat + '_' + f
    feat_data.rename(columns=nameMap, inplace=True)
    feat_data.reset_index(inplace=True)
    newFeats = [feat + '_' + f for f in feats]
    feat_data[newFeats].fillna(0, inplace=True)

    train = pd.merge(train, feat_data, on=[feat, '交易月份'])
    test = pd.merge(test, feat_data, on=[feat, '交易月份'])

    train.drop(feats, axis=1, inplace=True)
    test.drop(feats, axis=1, inplace=True)
    return train, test


def getRegionData(train, test):
    """
    计算行政区域的相关特征（对应板块的各个特征）
    """
    region_data = train.groupby(['region', '交易月份', 'plate']).first() \
        .groupby(['region', '交易月份']).sum()
    feats = ['totalTradeMoney', 'totalTradeArea', 'tradeMeanPrice',
             'tradeSecNum', 'totalNewTradeMoney', 'totalNewTradeArea',
             'tradeNewMeanPrice', 'tradeNewNum', 'remainNewNum']
    region_data = region_data[feats]
    nameMap = {}
    for f in feats:
        nameMap[f] = 'region_' + f
    region_data.rename(columns=nameMap, inplace=True)
    region_data.reset_index(inplace=True)
    newFeats = ['region_' + f for f in feats]
    region_data[newFeats].fillna(0, inplace=True)

    train = pd.merge(train, region_data, on=['region', '交易月份'], how='left')
    test = pd.merge(test, region_data, on=['region', '交易月份'], how='left')
    train['region'] = train['region'].astype('category')
    test['region'] = test['region'].astype('category')

    return train, test


def getStatisticalFeats(df_train, df_test):
    features = [_ for _ in df_train.columns if type(df_train[_].iloc[0]) != str]
    features.remove('tradeMoney')

    idx = features
    for df in [df_test, df_train]:
        df['mean'] = df[idx].mean(axis=1)
        df['std'] = df[idx].std(axis=1)
        df['skew'] = df[idx].skew(axis=1)
        df['kurt'] = df[idx].kurtosis(axis=1)
        df['med'] = df[idx].median(axis=1)

    return df_train, df_test


def getTotalSecNum(df_train, df_test):
    """
    计算板块二手房房源挂牌总数
    """
    train = df_train.copy()
    test = df_test.copy()

    data = train.copy()
    data.drop_duplicates('communityName', inplace=True)
    data = data[['plate', 'communityName', 'saleSecHouseNum']]
    data = data.groupby(['plate']).sum()
    data.reset_index(inplace=True)
    data.rename(columns={'saleSecHouseNum': 'totalSecNum'}, inplace=True)

    train = pd.merge(train, data, on=['plate'], how='left')
    test = pd.merge(test, data, on=['plate'], how='left')

    train['plate'] = train['plate'].astype('category')
    test['plate'] = test['plate'].astype('category')

    return train, test


def parseLandData(df_train, df_test):
    """
    计算行政区域区域的土地特征(与板块中的特征对应)
    """
    print("Parsing land data...", end='\t')
    train = df_train.copy()
    test = df_test.copy()

    data = train.groupby(['region', 'plate', '交易月份']).first()
    cols = ['supplyLandNum', 'supplyLandArea', 'tradeLandNum', 'tradeLandArea', 'landTotalPrice', 'landMeanPrice']
    data2 = data[cols].groupby(['region']).sum()
    newCols_map = {col: col + '_sum' for col in cols}
    data2.rename(columns=newCols_map, inplace=True)
    data2.reset_index(inplace=True)

    train = pd.merge(train, data2, on=['region'], how='left')
    test = pd.merge(test, data2, on=['region'], how='left')
    train['region'] = train['region'].astype('category')
    test['region'] = test['region'].astype('category')
    #train.drop(cols, axis=1, inplace=True)
    #test.drop(cols, axis=1, inplace=True)
    print('done\n')
    return train, test