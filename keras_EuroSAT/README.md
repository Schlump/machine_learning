# Classifiying Sentinel-2 Data on a CNN
## Train on 21600 samples, validate on 5400 samples (80/20)

### Data Description:
#### 10 Classes
#### 27000 x 64x64x13

| Number    | Type          |m²   |    nm |
|-----------|---------------|-----|-------|
B0 			|Aerosols		| 60  |	443   |
B1 			|Blue	 		| 10  |	490     |
B2 		 	|Green	 		| 10  |	560     |
B3 		 	|Red	 	 	| 10  |	665     |
B4 		 	|Red edge 1		| 20  |	705     |
B5 		 	|Red edge 2		| 20  |	740     |
B6 		 	|Red edge 3		| 20  |	783     |
B7 		 	|NIR	 		| 10  | 842     |
B8 			|Red edge 4 	| 20  | 865     |
B9 		 	|Water vapor 	| 60  |	945     |
B10  		|Cirrus 		| 60  |	1375    |
B11  		|SWIR 1 		| 20  |	1610    |
B12  		|SWIR 2 		| 20  |	2190    |




### Currently: 
| Data          | Accuracy      |Epochs |
| ------------- |-------------  |-------|
| RGB           | 0.919         | 205   |





