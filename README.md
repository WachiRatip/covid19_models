# covid19_models
โปรเจคนี้เป็นการสร้างโมเดลเพื่อทำนายจำนวนผู้ป่วย covid19 โดยใช้วิธีที่แตกต่างกันจาก 3 ศาสตร์คือ 
* โมเดลทางคณิตศาสตร์(ในที่นี้คือ SEIR model) 
* โมเดลทางการเรียนรู้ของเครื่อง(ในที่นี้คือ LSTMs) 
* และโมเดลทางสถิติ(ในที่นี้คือ ARIMA)

ข้อมูลที่ใช้งานอ้างอิงจากข้อมูลของ [Johns Hopkins University Center for Systems Science and Engineering (JHU CSSE)](https://github.com/CSSEGISandData/COVID-19?fbclid=IwAR1CgDWM6JCC7QHLwpGj24LRODW50YWd6DUNGhYVIP90bevwgX_z1EUM0yY)

# แนวคิดในการสร้างโมเดล(อย่างย่อ)

1. มี 3 โมเดลคือ SEIR, LSTMs, และ ARIMA
2. ทำการทำนายสองแบบคือ   
   ทำนายอนาคตตั้งแต่วันนี้ไปโดยใช้ข้อมูลอดีตถึงปัจจุบันเป็นข้อมูลฝึกฝน และ   
   ทำนายเหตุในช่วงเวลาสองสัปดาห์ล่าสุดโดยใช้ข้อมูลอดีตของสองสัปดาห์ที่แล้วเป็นข้อมูลฝึกฝน (validation)   
3. การทำนายทั้งหมดอยู่บนสมมติฐานที่ว่า
   การระบาดสามารถไปได้ทั่วโลก   
   จำนวนผู้ป่วยที่ยืนยัน (confirmed cases) แล้วคือผู้ป่วยที่ติดเชื้อและแสดงอาการทั้งหมด (infected group)   
   จำนวนผู้ติดเชื้อแต่ยังไม่แสดงอาการ (exposed group) มีจำนวนเป็น 10 เท่าของผู้ป่วย   
   โมเดลจะทำนายผู้ป่วยสะสม (Total infected cases) ในแต่ละวัน   
   โรคนี้ติดจากคนสู่คนเท่านั้น   
   คนที่รักษาหายแล้วจะไม่ป่วยอีก   
   จำนวนผู้ป่วย (infected) ในแต่ละวันคำนวณจาก <ผู้ติดเชื้อยืนยัน (confirmed cases) - จำนวนผู้ที่รักษาหาย (recovered cases) - จำนวนผู้เสียชีวิต (death cases)>   
##### SEIR model
1. ระหว่างนี้ให้ถือว่าไม่มีการเกิดและไม่มีการตายของมนุษย์โลก
2. จำนวน N ให้มีค่าเท่ากับจำนวนประชากรโลกในวันนี้ [อ้างอิง](https://www.worldometers.info/world-population/) 
3. ค่าตั้งต้น I(0) ให้มีค่าเท่ากับ <จำนวนผู้ป่วยที่ยืนยันแล้ว - ผู้เสียชีวิต - ผู้ที่รักษาหายแล้ว> ในคอลัมน์สุดท้ายของ time-series dataset
4. ค่าตั้งต้น E(0) ให้มีค่าเท่าเป็นสิบเท่าของ I(0)
5. ค่าตั้งต้น R(0) ให้มีค่าเท่ากับ <ผู้เสียชีวิต + ผู้ที่รักษาหายแล้ว> ในคอลัมน์สุดท้ายของ time-series dataset นั้นคือคนตายอยู่ในกลุ่มคนที่รักษาหายแล้วและมีภูมิต้านทานโรค
6. ค่าตั้งต้น S(0) = N - E(0) - I(0) - R(0)
7. recovery rate ให้คำนวณจากค่าเฉลี่ยของการเปลี่ยนแปลงจำนวนผู้ที่รักษาหายแล้วและผู้เสียชีวิตในแต่ละวันบวกกัน
8. incubation rate ให้คำนวณจากค่าเฉลี่ยของการเปลี่ยนแปลงจำนวนผู้ติดเชื้อยืนยันแล้วในแต่ละวัน
9. contact rate ให้คำนวณโดยใช้สูตร contact rate = reproductive number * reconvery rate
10. สมมติให้ reproductive number = 2 ตาม [john hopkins university](https://systems.jhu.edu/research/public-health/ncov-model/)
##### LSTMs model
1. Training set สร้างโดยตัด time-series หนึ่งอันออกมาเป็น time-series ย่อย ๆ ที่มีความยาว 5 วันต่อหนึ่ง time-series (ใช้วิธีตัดที่วันที่ 5 แล้วเลื่อน 1 วันแล้วตัดอีกครั้งในวันที่ 5 (นั้นคือวันที่ 6 ใน time-series ใหญ่) แล้ววนลูปจนหมด)
2. ในการ validation 14 วัน จะใช้ time-series dataset ของ 14 วันที่แล้วเป็นชุดฝึกฝน
3. สร้างโมเดลแยกกันเพื่อทำนายจำนวนผู้ป่วยยืนยันแล้ว (confirmed) ผู้เสียชีวิต (death) ผู้ที่รักษาหายแล้ว (recovered)
4. โมเดลสุดท้ายจะทำนายจำนวนผู้ป่วยสะสมในแต่ละวันโดยคำนวนจาก <ค่าทำนายผู้ป่วยที่ยืนยันแล้ว - ค่าทำนายผู้เสียชีวิต - ค่าทำนายผู้รักษาหายแล้ว> ในแต่ละวัน
##### ARIMA model
1. ใช้ time-series ทั้งหมดเป็น training set โดยไม่ตัดออก
2. ในการ validation 14 วัน จะใช้ time-series dataset ของ 14 วันที่แล้วเป็นชุดฝึกฝน
3. กำหนดค่านัยสำคัญทางสถิติที่ 0.05
4. สร้างโมเดลแยกกันเพื่อทำนายจำนวนผู้ป่วยยืนยันแล้ว (confirmed) ผู้เสียชีวิต (death) ผู้ที่รักษาหายแล้ว (recovered)
5. โมเดลสุดท้ายจะทำนายจำนวนผู้ป่วยสะสมในแต่ละวันโดยคำนวนจาก <ค่าทำนายผู้ป่วยที่ยืนยันแล้ว - ค่าทำนายผู้เสียชีวิต - ค่าทำนายผู้รักษาหายแล้ว> ในแต่ละวัน

# How to use it
การใช้งานไฟล์เหล่านี้สามารถใช้ได้หลายวิธีด้วยกันขึ้นอยู่กับความถนัดของแต่ละคน จะใช้งานผ่าน dockerfile หรือผ่าน python3 บนเครื่องโดยตรงก็ได้ หรือใช้งานผ่าน Google Colab (จะตามมาเร็ว ๆ นี้)

### ใช้งานผ่าน Docker container
##### สำหรับผู้ใช้ linux
1. ติดตั้ง [Docker CE](https://docs.docker.com/install/linux/docker-ce/ubuntu/) ตาม Linux distribution ที่ใช้งาน
2. เปิด Terminal แล้วย้าย Directory ไปที่ ๆ dockerfile อยู่ แล้วใช้คำสั่ง sudo bash build.sh start
3. เชื่อมต่อผ่าน juyther notebook ผ่าน link ที่แสดงใน Terminal หรือเข้าผ่าน VScode ด้วย [extension Remote - Containers](https://code.visualstudio.com/docs/remote/containers)
4. เมื่อเลิกใช้งานให้กด ctrl+C ขณะเปิดหน้า Terminal
##### สำหรับผู้ใช้ Windows หรือ MacOS
1. ติดตั้ง [Docker Desktop](https://www.docker.com/products/docker-desktop) และเปิดใช้งาน Docker Desktop
2. เปิด Terminal(MacOS) หรือ Power Shell(Windows) แล้วย้าย Directory ไปที่ ๆ dockerfile อยู่
3. ใช้คำสั่ง docker build -t covid19_model:1.0.0 . เพื่อสร้าง Docker Image
4. ใช้คำสั่ง docker run -p 8888:8888 --name container_covid -v $(pwd)/scr:/home:rw covid19_model:1.0.0 เพื่อสร้าง Container (สังเกตว่า covid_model:1.0.0 จะตรงกับตอนสร้าง Docker Image)
5. เชื่อมต่อผ่าน juyther notebook ผ่าน link ที่แสดงใน Terminal หรือเข้าผ่าน VScode ด้วย [extension Remote - Containers](https://code.visualstudio.com/docs/remote/containers)
6. เมื่อเลิกใช้งานให้กด ctrl+C ขณะเปิดหน้า Terminal หรือ Power Shell

### ใช้งานผ่านการรัน python code โดยตรง
##### สำหรับผู้ใช้ linux
1. ติดตั้ง python version 3.6.9 ขึ้นไป แต่โดยทั่วไป linux มักติดตั้ง python3 มาให้อยู่แล้ว (พิมพ์ python3 --version หรือ python --version เพื่อทดสอบ)
2. ติดตั้ง packages ที่จำเป็นทั้งหมดคือ pytoch, torchvision, numpy, pandas, scipy, matplotlib, joblib, scikit-learn, และ statsmodels   
   หรือใช้คำสั่ง pip3 install requirements.txt หรือ pip install requirements.txt (ต้องติดตั้ง pip/pip3 ก่อนหากยังไม่ติดตั้ง)
3. เปิดไฟล์ main.py หรือ main.ipynb เพื่อรันโปรแกรมผ่าน editor/ide ที่ถนัดได้เลย
##### สำหรับผู้ใช้ Windows หรือ MacOS
1. ติดตั้ง python version 3.6.9 แนะนำ [Anaconda Distribution](https://www.anaconda.com/distribution/)
2. ติดตั้ง packages ที่จำเป็นทั้งหมดคือ pytoch, torchvision, numpy, pandas, scipy, matplotlib, joblib, scikit-learn, และ statsmodels 
   หรือใช้คำสั่ง pip3 install requirements.txt หรือ pip install requirements.txt (Note: แก้ไขไฟล์ requirements.txt ในส่วนของ pytorch ให้เป็นลิงค์ตาม OS ที่ใช้งาน [ดูชื่อ link](https://download.pytorch.org/whl/cpu/torch_stable.html))
3. เปิดไฟล์ main.py หรือ main.ipynb เพื่อรันโปรแกรมผ่าน editor/ide ที่ถนัดได้เลย

### ใช้งานผ่าน Google Colab
###### [Google Colab](https://colab.research.google.com/drive/1mLQAKkSjh2RcQ7fKAC8OzfM6P4SqhMWG?fbclid=IwAR3ZD5Zl-4bw_ECZKLXqySlq6re-RBsipjJh9muKdFmKVcSUB1h9yu1MLBE)(แบบแก้ไขได้จะตามมาเร็ว ๆ นี้)
