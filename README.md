Code for Analytics Genpact competititon

**Steps to reporduce solution for 3rd place**:
* Change file locations in config file
* run `bash run_all.sh`



**Dependencies**:

    python == 3.6.4
    numpy == 1.14.2
    pandas == 0.23.4
    sklearn == 0.20.0
    implicit == 0.3.8
    tqdm == 4.23.4    
    pathlib
    
**Approach**:

#### Validation strategy
Keep last 2 months data as validation set mimicking train/test split  

#### Feature engineering
   * As the solution was based on weighted matrix factorization, it was important to get right weights.
   I started by taking sum of item quantity ordered. Soon, I realized time played important role, so I decided to inversely weigh quantities with respect to time. FInally, I also added time weighting w.r.t overall product purchase. 

#### Models
   * BM25 recommender from implicit library

#### Key takeaway:
   * It was important to use right weighting for user-item matrix
 
#### 5 things to focus
   * Validation strategy - split train and validation sets in time, mimick train/test
   * Trying lot of models - I tried lot of matrix factorization and finally settled on BM25 recomnder from implicit
   * Ensemble doesn't always work - I tried emsembling scores from different MF models but it made results worse on validation
   * Hyperparameter tuning - It was really crucial
   * Don't trust things as is - There was sort of bug in helper script where userid, productid were used as row, column to create sparse matrices. But productids were notnumbered 0-n which caused matrix dimension to increase and a huge running time. Took me a while to notice
   
Thanks for reading :)
 
 

