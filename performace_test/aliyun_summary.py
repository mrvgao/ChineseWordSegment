import requests


host = 'http://wbzy.market.alicloudapi.com'
path = '/rest/160601/text_analysis/attention_text_summary.json'
method = 'POST'
appcode = '4b3979c17416404e9b901bce0262ad3a'
querys = ''
bodys = {}
url = host + path

text = "“像家一样”,这样的酒店宣传语估计要大打折扣了,理由就是——脏!日前,加拿大一项堪称最大规模的调查显示,包括速8、喜来登以及假日酒店在内的6大连锁酒店用品,如床上用品、遥控器等都有细菌,威胁住客健康,希望连锁酒店给住客一个真正干净的“家”"

headers = {'Authorization': 'APPCODE ' + appcode,
           'Content-Type': 'application/json; charset=UTF-8'}

data = """{
  "inputs": [
    {
      "text": {
        "dataType": 50,
        "dataValue": text
      }
    }
  ]
}"""
r = requests.post(host+path, data=data, headers=headers)

content = r.text

print(content)
