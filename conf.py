# As of 4/01/2017
databases = [
     # Arrhythmia
    ('mitdb', ["100", "101", "102", "103", "104", "105", "106", "107", "108", "109", "111", "112", "113", "114", "115", "116", "117", "118", "119", "121", "122", "123", "124", "200", "201", "202", "203", "205", "207", "208", "209", "210", "212", "213", "214", "215", "217", "219", "220", "221", "222", "223", "228", "230", "231", "232", "233", "234"]),
    # Long Term
    ('ltdb', ["14046", "14134", "14149", "14157", "14172", "14184", "15814"]),
    # Normal Sinus Rhythm (Changing leads)
    ('nsrdb', ["16265", "16272", "16273", "16420", "16483", "16539", "16539", "16773", "16773", "16786", "16795", "17052", "17453", "18177", "18184", "19088", "19090", "19093", "19140", "19830"]), 
    # Supraventricular Arrhythmia
    ('svdb', ["801", "801", "802", "806", "807", "808", "811", "821", "824", "825", "826", "827", "828", "828", "829", "829", "840", "841", "841", "842", "843", "844", "845", "846", "849", "850", "852", "852", "859", "864", "864", "866", "867", "869", "871", "872", "873", "874", "875", "876", "876", "877", "877", "880", "880", "882", "883", "886", "889", "890", "893", "800", "803", "804", "805", "809", "810", "812", "820", "822", "823", "847", "848", "851", "853", "854", "854", "855", "856", "856", "857", "858", "860", "861", "862", "863", "865", "868", "870", "878", "879", "881", "884", "885", "887", "888", "891", "892", "894"]),
    # European ST-T (Beats on non-R)
    ('edb', ["e0103", "e0104", "e0105", "e0106", "e0107", "e0108", "e0110", "e0111", "e0112", "e0113", "e0114", "e0115", "e0116", "e0118", "e0119", "e0121", "e0122", "e0123", "e0124", "e0125", "e0126", "e0127", "e0129", "e0133", "e0136", "e0139", "e0147", "e0148", "e0151", "e0154", "e0155", "e0159", "e0161", "e0162", "e0163", "e0166", "e0170", "e0202", "e0203", "e0205", "e0206", "e0207", "e0208", "e0210", "e0211", "e0212", "e0213", "e0302", "e0303", "e0304", "e0305", "e0306", "e0403", "e0404", "e0405", "e0406", "e0408", "e0409", "e0410", "e0411", "e0413", "e0415", "e0417", "e0418", "e0501", "e0509", "e0515", "e0601", "e0602", "e0603", "e0604", "e0605", "e0606", "e0607", "e0609", "e0610", "e0611", "e0612", "e0613", "e0614", "e0615", "e0704", "e0801", "e0808", "e0817", "e0818", "e1301", "e1302", "e1304"]), #, "e0204"
    # Long-Term AF (Changing leads)
    ('ltafdb', ["00", "01", "03", "05", "06", "07", "08", "10", "100", "101", "102", "103", "104", "105", "11", "110", "111", "112", "113", "114", "115", "116", "117", "118", "119", "12", "120", "121", "122", "13", "15", "16", "17", "18", "19", "20", "200", "201", "202", "03", "204", "205", "206", "207", "208", "21", "22", "23", "24", "25", "26", "28", "30", "32", "33", "34", "35", "37", "38", "39", "42", "43", "44", "45", "47", "48", "49", "51", "53", "54", "55", "56", "58", "60", "62", "64", "65", "68", "69", "70", "71", "72", "74", "75"]),
    # Creighton University Ventricular Tachyarrhythmia (No peak labels)
    ('cudb', ['cu01', 'cu02', 'cu03', 'cu04', 'cu05', 'cu06', 'cu07', 'cu08', 'cu09', 'cu10', 'cu11', 'cu12', 'cu13', 'cu14', 'cu15', 'cu16', 'cu17', 'cu18', 'cu19', 'cu20', 'cu21', 'cu22', 'cu23', 'cu24', 'cu25', 'cu26', 'cu27', 'cu28', 'cu29', 'cu30', 'cu31', 'cu32', 'cu33', 'cu34', 'cu35']),
    # Arrhythmia Database (beat annotations not always avail)
    ('incartdb', ['I01', 'I02', 'I03', 'I04', 'I05', 'I06', 'I07', 'I08', 'I09', 'I10', 'I11', 'I12', 'I13', 'I14', 'I15', 'I16', 'I17', 'I18', 'I19', 'I20', 'I21', 'I22', 'I23', 'I24', 'I25', 'I26', 'I27', 'I28', 'I29', 'I30', 'I31', 'I32', 'I33', 'I34', 'I35', 'I36', 'I37', 'I38', 'I39', 'I40', 'I41', 'I42', 'I43', 'I44', 'I45', 'I46', 'I47', 'I48', 'I49', 'I50', 'I51', 'I52', 'I53', 'I54', 'I55', 'I56', 'I57', 'I58', 'I59', 'I60', 'I61', 'I62', 'I63', 'I64', 'I65', 'I66', 'I67', 'I68', 'I69', 'I70', 'I71', 'I72', 'I73', 'I74', 'I75']),
    # Changing leads (beat annotations not always avail)
    ('sddb', ['30', '31', '32', '34', '35', '36', '41', '45', '46', '51', '52']), # '49'
    ('ltstdb', ['s20011', 's20021', 's20031', 's20041', 's20051', 's20061', 's20071', 's20081', 's20091', 's20101', 's20111', 's20121', 's20131', 's20141', 's20151', 's20161', 's20171', 's20181', 's20191', 's20201', 's20211', 's20221', 's20231', 's20241', 's20251', 's20261', 's20271', 's20272', 's20273', 's20274', 's20281', 's20291', 's20301', 's20311', 's20321', 's20331', 's20341', 's20351', 's20361', 's20371', 's20381', 's20391', 's20401', 's20411', 's20421', 's20431', 's20441', 's20451', 's20461', 's20471', 's20481', 's20491', 's20501', 's20511', 's20521', 's20531', 's20541', 's20551', 's20561', 's20571', 's20581', 's20591', 's20601', 's20611', 's20621', 's20631', 's20641', 's20651', 's30661', 's30671', 's30681', 's30691', 's30701', 's30711', 's30721', 's30731', 's30732', 's30741', 's30742', 's30751', 's30752', 's30761', 's30771', 's30781', 's30791', 's30801']),
    ('stdb', ['300', '301', '302', '303', '304', '305', '306', '307', '308', '309', '310', '311', '312', '313', '314', '316', '317', '318', '319', '320', '321', '322', '323', '324', '325', '326', '327']), # '315'
    ('challenge/2014/set-p2', ['1009', '1016', '1019', '1020', '1022', '1023', '1028', '1032', '1033', '1036', '1043', '1069', '1071', '1073', '1077', '1169', '1195', '1242', '1284', '1354', '1376', '1388', '1447', '1456', '1485', '1503', '1522', '1565', '1584', '1683', '1686', '1715', '1742', '1774', '1804', '1807', '1821', '1858', '1866', '1900', '1906', '1954', '1993', '1998', '2041', '2063', '2132', '2164', '2174', '2201', '2203', '2209', '2247', '2277', '2279', '2283', '2296', '2327', '2370', '2384', '2397', '2469', '2527', '2552', '2556', '2602', '2639', '2664', '2714', '2728', '2732', '2733', '2798', '2800', '2812', '2839', '2850', '2879', '2885', '2886', '2907', '2923', '2970', '3188', '3266', '41024', '41025', '41081', '41164', '41173', '41180', '41566', '41778', '41951', '42228', '42511', '42878', '42961', '43247']),
    # NO BEAT AVAILABLE BEHIND
    # Atrial Fibrillation (no beat annotation avail.)
    #('afdb', ["04015", "04043", "04048", "04126", "04746", "04908", "04936", "05091", "05121", "05261", "06426", "06453", "06995", "07162", "07859", "07879", "07910", "08215", "08219", "08378", "08405", "08434", "08455"]),
    # Malignant Ventricular Arrhythmia (no beat annotation avail.)
    #('vfdb', ["418", "419", "420", "421", "422", "423", "424", "425", "426", "427", "428", "429", "430", "602", "605", "607", "609", "610", "611", "612", "614", "615"]),
]

Normal_beat = 'N'
beat_annotations = ['N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r', 'F', 'e', 'j', 'n', 'E', '/', 'f', 'Q', '?']
non_beat_annotations = ['[', '!', ']', 'x', '(', ')', 'p', 't', 'u', '`', '\'', '^', '|', '~', '+', 's', 'T', '*', 'D', '=', '"', '@']
url = "https://physionet.org/physiobank/database/"
data_dir = 'data/'



architecture_1 = [
    (0.5, [
            (3, 8, 'rectify', 'same'),
            (15, 64, 'rectify', 'same'),
            (45, 64, 'rectify', 'same'),
            (89, 32, 'rectify', 'same'),
            (149, 16, 'rectify', 'same'),
            (199, 16, 'rectify', 'same'),
            (299, 16, 'rectify', 'same'),
    ]),
    (0.5, [
            (3, 8, rectify, 'same'),
            (9, 64, rectify, 'same'),
            (19, 64, rectify, 'same'),
            (39, 64, rectify, 'same'),
    ]),
    (0.5, [
            (3, 8, rectify, 'same'),
            (5, 32, rectify, 'same'),
            (9, 32, rectify, 'same'),
            (15, 32, rectify, 'same'),
            (19, 32, rectify, 'same'),
    ]),
    (0.5, [
            (3, 8, rectify, 'same'),
            (5, 32, rectify, 'same'),
            (15, 64, rectify, 'same'),
    ]),
    (0.5, [
            (5, 16, rectify, 'same'),
            (15, 16, rectify, 'same'),
            (19, 16, rectify, 'same'),
    ]),
    (0.5, [
            (3, 8, rectify, 'same'),
            (9, 16, rectify, 'same'),
            (15, 16, rectify, 'same'),
    ])
]
