c = get_config()

c.InteractiveShellApp.exec_lines = [
        "import pandas as pd",
        "import numpy as np",
        "import matplotlib.pyplot as plt", 
		"plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签",
		"plt.rcParams['axes.unicode_minus']=False #用来正常显示负号",
		'%autoreload 2',
        ]
		
c.IPKernelApp.matplotlib = 'inline'
c.InteractiveShellApp.extensions = ['autoreload']     
c.InlineBackend.figure_format = 'retina'



c.NotebookApp.certfile = 'E:\学习\code\others\jupyter\mycert.pem'
c.NotebookApp.ip = '*'
c.NotebookApp.password = 'sha1:2ade213e9694:45d57f8f5d6b5032d985fb9f4693913920ff2bb2'
c.NotebookApp.port = 9999