# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""script for collecting mmcif download link"""

import os
import stat
import requests
from lxml import etree

if __name__ == "__main__":
    URL = 'https://ftp.rcsb.org/pub/pdb/data/structures/divided/mmCIF/'
    body = requests.get(URL)
    body.encoding = 'gbk'
    FLAG = os.O_RDWR | os.O_CREAT
    MODE = stat.S_IRWXU

    html = etree.HTML(body.text)
    divs = html.xpath('//pre/a/text()')
    for url_ in divs[4:]:
        url_2 = URL+url_
        body_ = requests.get(url_2)
        body_.encoding = 'gbk'

        html_ = etree.HTML(body_.text)
        data_urls = html_.xpath('//pre/a/text()')[4:]

        for i in data_urls:
            with os.fdopen(os.open('file_download_link.txt', FLAGS, MODES), 'w') as fout:
                data_url = url_2+i
                fout.write(data_url+'\n')
                print(data_url)
