import im2data as im
import Non_linear_method as nlm

name = "rabbit"
mvt_num = 6

im.im2data(name, mvt_num)
res = nlm.nlmethod(name + "_Mvt_" + str(mvt_num), (5000, 0.001))

nlm.plot_result(res)
