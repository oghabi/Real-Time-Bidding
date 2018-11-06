
## Real-Time Bidding in Computational Advertising
The goal is to devise a bidding strategy in order for advertising campaigns to place their ads online (ie. submit them to an ad exchange) under a strict budget. The bidding strategy is devised by predicting the CTR (click-through rate) of ads using a machine learning model.
Some research papers on Real-Time bidding can be found [here](https://github.com/wnzhang/rtb-papers)

## Dataset
The dataset is the [iPinYou Dataset](http://data.computational-advertising.org/). It is a CSV format and an example is shown below. Note that the ‘bidprice’, ‘payprice’ and ‘click’ columns are not available in the test set.

|Field  |Example|Supplement|
|-------|-------|-------|
|click |1 |1 if clicked, 0 if not.
|weekday |1
|hour |12
|bidid |fdfe...b8b21
|logtype |1
|userid |u_Vh1OPkFv3q5CFdR
|useragent |windows_ie
|IP |180.107.112.*
|region |80
|city |85
|adexchange |2
|domain |trq...Mi
|url |d48a...4efeb
|urlid |as3d...34frg
|slotid |2147...813
|slotwidth |300
|slotheight |250
|slotvisibility |SecondView
|slotformat |Fixed
|slotprice |0
|creative |hd2v...jhs72
|bidprice |399
|payprice |322 |Paid price afterwin the bidding.
|keypage |sasd...47hsd
|advertiser |2345
|usertag |123,5678,3456 |Contains multivalues,‘,’ as segmentation.


## Results
[Report 2](https://github.com/oghabi/Real-Time-Bidding/blob/master/Report%201.pdf) reports the statistics and an analysis of the dataset. It also compares different machine learning CTR (click-through rate) prediction models and compares linear and non-linear bidding strategies. The evaluations are done on the test set.

[Report 1](https://github.com/oghabi/Real-Time-Bidding/blob/master/Report%202.pdf) reports a more complex CTR prediction model and bidding strategy which yields better results. The evaluations are done on the test set.
