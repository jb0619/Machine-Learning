

 ****題目:Maximum A Posteriori Estimation


使用Maximum A Posteriori(MAP)方式從60個測試樣本中區分3種類型的酒。

 ****模組套件

        Python 3
	Python module
	pandas
	NumPy
	scikit-learn
	Matplotlib
 
 ****如何執行

可以使用jupyter-notebook打開"HW1.ipynb"且按下Run去執行程式碼。

 ****討論

在算likelihood時，我把原本的數值取對數，目的為了使的數據正規化，避免各特徵數據差距太大導致數值underflow。