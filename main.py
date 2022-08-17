from operator import truediv
import os, io
from google.cloud import vision
from google.cloud.vision_v1 import types
import pandas as pd


os.environ['GOOGLE_APPLICATION_CREDENTIALS']= r'winter-justice-358511-4035a6838b34.json'

client = vision.ImageAnnotatorClient()

trainingData = 'LIFE TO BE ASSURED / ORANG YANG AKAN DIINSURANSKAN\nNRIC number (new) / Nombor kad pengenalan (baharu)\n2 2 1 2 2 1\n32\nOther identification number /\nNombor pengenalan lain\nDate of birth / Tarikh lahir\n21\n01\nGender / Jantina\nMale /\n✓ Lelaki\nMarital status/Status perkahwinan\nMarried /\nBerkahwin\nMalay /\nMelayu\nCountry / Negara\nFemale /\nPerempuan\nSingle /\nBujang\nRace / Kaum\n✓\nNationality / Kewarganegaraan\nMalaysian /\nOthers /\nMalaysia\nLain-lain\nCorrespondence address / Alamat surat-menyurat\nIn Malaysia\nCountry / Negara\n1970\nChinese /\nCina\nDivorced /\nBercerai\n0 1 1\nEmail / Emel\nIndian /\nIndia\nPostcode /\nPoskod\n1 2 3 4\nMalaysia\nResidential address (if different from correspondence address) /\nAlamat rumah (jika lain dari alamat surat-menyurat)\nPostcode /\nPoskod\nOccupation / Pekerjaan\nJob\nWidowed /\nBalu\nOthers /\nLain-lain\nPhone number / Nombor telefon\na. Mobile telephone number / Nombor telefon bimbit\n012\n3456 789\n1 2 3 4 5\nb. Residence telephone number / Nombor telefon kediaman\n0 1 1\n1 2 3 4 5 6 7 8\nc. Office telephone number / Nombor telefon pejabat\n1 2 3 4 5 6 7 8\nemasil@test c o m\nPROPOSER (if different from life to be assured) /\nPENCADANG (jika lain dari orang yang akan diinsuranskan)\nNRIC number (new) / Nombor kad pengenalan (baharu)\nOther identification number/Business registration number /\nNombor pengenalan lain/Nombor pendaftaran perniagaan\nDate of birth / Tarikh lahir\nD\nM M\nGender / Jantina\nMale /\nLelaki\nSingle /\nBujang\nRace / Kaum\nMalay /\nMelayu\nMarital status/Status perkahwinan\nMarried /\nBerkahwin\nCountry / Negara\nCountry / Negara\nFemale /\nPerempuan\nY\nChinese /\nCina\nNationality / Kewarganegaraan\nMalaysian /\nMalaysia\nCorrespondence address / Alamat surat-menyurat\nEmail / Emel\nY\nOthers/\nLain-lain\n2/15\nY Y\nDivorced /\nBercerai\nIndian/\nIndia\nOccupation / Pekerjaan\nResidential address (if different from correspondence address) /\nAlamat rumah (jika lain dari alamat surat-menyurat)\nPostcode /\nPoskod\nPhone number / Nombor telefon\na. Mobile telephone number / Nombor telefon bimbit\nPostcode /\nPoskod\nOthers /\nLain-lain\nb. Residence telephone number / Nombor telefon kediaman\nc. Office telephone number / Nombor telefon pejabat\nWidowed /\nBalu\nYour policy/certificate will be sent to your email address stated above. You may also refer to your policy/certificate details anytime via our client portal\nsunaccess.sunlifemalaysia.com/portal-ui/CUSTOMER/login. If there is no valid email address provided, your policy/certificate will be mailed to your correspondence\naddress. / Polisi/sijil anda akan dihantar kepada emel anda seperti di atas. Anda juga boleh merujuk kepada butiran polisi/sijil anda pada bila-bila masa melalui\nsunaccess.sunlifemalaysia.com/portal-ui/CUSTOMER/login. Jika tiada emel sah diberi, polisi/sijil anda akan dihantar ke alamat surat-menyurat anda.\nPlease select the preferred language for your policy/certificate. / Sila pilih bahasa yang dikehendaki untuk polisi/sijil anda.\nEnglish / Bahasa Inggeris\nMalay / Bahasa Melayu\nIf there is no preference selected, your policy/certificate will be in English. / Jika tiada pilihan dibuat, Bahasa Inggeris akan digunakan untuk polisi/sijil anda.\nMonthly income / Pendapatan bulanan\nMonthly income / Pendapatan bulanan\nR M2000\nR M'
mobileSavedToPDFImage = 'D:/DocOCR/ocr_analysis/trainImage.jpg'
cameraImage = 'D:/DocOCR/ocr_analysis/testImage.jpg'

# Data processed from Google Vision API for PDF Files
# Script is in dococr_backend googlevisionapi
pdfData = "LIFE TO BE ASSURED / ORANG YANG AKAN DIINSURANSKAN\nNRIC number (new) / Nombor kad pengenalan (baharu)\n221221-3 2-1 2 3 4\nOther identification number /\nNombor pengenalan lain\nDate of birth / Tarikh lahir\n21-01-1970\nGender/Jantina\nMale/\nLelaki\nMarital status/Status perkahwinan\nMarried /\nBerkahwin\nSingle/\nBujang\nRace / Kaum\nMalay/\nMelayu\nFemale/\nPerempuan\nCountry / Negara\nChinese/\nCina\nNationality/Kewarganegaraan\nMalaysian/\nMalaysia\nCorrespondence address/Alamat surat-menyurat\nIn Malaysia\nEmail / Emel\nDivorced /\nBercerai\nOthers/\nLain-lain\nIndian/\nIndia\nPostcode/\nPoskod\nCountry / Negara\nMalaysia\nResidential address (if different from correspondence address) /\nAlamat rumah (jika lain dari alamat surat-menyurat)\nOthers/\nLain-lain\nPostcode/\nPoskod\nOccupation / Pekerjaan\nJob\nWidowed/\nBalu\nPhone number / Nombor telefon\na. Mobile telephone number / Nombor telefon bimbit\n1 2 3 4 5\n012-3456789\nb. Residence telephone number / Nombor telefon kediaman\n011- 1 2 3 4 5 6 7 8\nc. Office telephone number / Nombor telefon pejabat\n011-12345678\nemasil@test.com\nPROPOSER (if different from life to be assured)/\nPENCADANG (jika lain dari orang yang akan diinsuranskan)\nNRIC number (new) / Nombor kad pengenalan (baharu)\nOther identification number/Business registration number /\nNombor pengenalan lain/Nombor pendaftaran perniagaan\nDate of birth/Tarikh lahir\nGender/Jantina\nMale/\nLelaki\nFemale/\nPerempuan\nMarital status/Status perkahwinan\nMarried/\nBerkahwin\nSingle/\nBujang\nRace / Kaum\nChinese/\nCina\nNationality / Kewarganegaraan\nMalaysian /\nMalaysia\nCorrespondence address/Alamat surat-menyurat\nMalay/\nMelayu\nCountry / Negara\nCountry / Negara\nDivorced/\nBercerai\nOthers/\nLain-lain\nEmail / Emel\nIndian/\nindia\nResidential address (if different from correspondence address) /\nAlamat rumah (jika lain dari alamat surat-menyurat)\nPostcode/\nPoskod\nPhone number / Nombor telefon\nMobile telephone number / Nombor telefon bimbit\nOccupation / Pekerjaan\n2/15\nPostcode/\nPoskod\nOthers/\nLain-lain\nb. Residence telephone number / Nombor telefon kediaman\nc. Office telephone number / Nombor telefon pejabat\nWidowed/\nBalu\nYour policy/certificate will be sent to your email address stated above. You may also refer to your policy/certificate details anytime via our client portal\nsunaccess.sunlifemalaysia.com/portal-ui/CUSTOMER/login. If there is no valid email address provided, your policy/certificate will be mailed to your correspondence\naddress. / Polisi/sijil anda akan dihantar kepada emel anda seperti di atas. Anda juga boleh merujuk kepada butiran polisi/sijil anda pada bila-bila masa melalui\nsunaccess.sunlifemalaysia.com/portalui/CUSTOMER/login. Jika tiada emel sah diberi, polisi/sijil anda akan dihantar ke alamat surat-menyurat anda.\nPlease select the preferred language for your policy/certificate./ Sila pilih bahasa yang dikehendaki untuk polisi/sijil anda.\nEnglish / Bahasa Inggeris\nMalay / Bahasa Melayu\nIf there is no preference selected, your policy/certificate will be in English./Jika tiada pilihan dibuat, Bahasa Inggeris akan digunakan untuk polisi/sijil anda\nMonthly income / Pendapatan bulanan\nMonthly income / Pendapatan bulanan\nRM2000\nR M"



def detect_text(path):
    """Detects text in the file."""
    from google.cloud import vision
    import io
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.full_text_annotation
    print('Texts:')
    # print(texts)
    return texts.text

mobileSavedToPDFData = detect_text(mobileSavedToPDFImage)
cameraData = detect_text(cameraImage)
'D:/DocOCR/ocr_analysis/testImage.jpg'
trainData = trainingData.split("\n")
mobileSavedToPDFData = mobileSavedToPDFData.split("\n")
cameraData = cameraData.split("\n")
pdfData = pdfData.split("\n")

# Initialise Dataframe
dictData = {'trainData':trainData,'From Mobile App PDF': mobileSavedToPDFData ,'Camera': cameraData, 'PDF': pdfData}
df =pd.DataFrame.from_dict(dictData, orient = 'index').to_csv('DataNew.csv')

# df = df.transpose

# print(len(trainData),len(cameraData))
print(df)