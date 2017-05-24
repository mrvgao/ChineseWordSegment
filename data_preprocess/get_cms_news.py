from bs4 import BeautifulSoup
import pandas as pd
import re
import random

def clearify(content, http_input=True):
    if http_input:
        soup = BeautifulSoup(content, 'html.parser')
        content = soup.get_text()
    content = content.strip()
    white_space_regex = re.compile(r'[\n\r\t\xa0@]')
    content = white_space_regex.sub("", content)
    quote_regx = re.compile(r'''['"、。，；;,!！“’”]''')
    content = quote_regx.sub(" ", content)
    content = re.sub( '\s+', ' ', content).strip()
    return content


def save_pure_content(new_file, pure_content_file, sample=1.):
    cms = pd.read_csv(new_file)

    databases = cms.iterrows()
    with open(pure_content_file, 'a') as f:
        for index, r in enumerate(databases):
            if index % 100 == 0:
                print(index)

            if random.random() > sample:
                continue

            try:
                content = r[1][2] + r[1][17]
                content = clearify(content)
                f.write(content + '\n')
            except TypeError:
                print(r)

if __name__ == '__main__':
    yesterday = 'updated_news/20170523.csv'
    content_file = 'updated_news/20170523-pure.data'
    save_pure_content(yesterday, pure_content_file=content_file, sample=0.5)
    print('done!')
