# deep-learning-visual-computing-assignment-1-linear-models-solved
**TO GET THIS SOLUTION VISIT:** [Deep Learning Visual Computing Assignment 1-Linear Models Solved](https://www.ankitcodinghub.com/product/deep-learning-visual-computing-assignment-1-linear-models-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;99773&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;3&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (3 votes)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;Deep Learning Visual Computing Assignment 1-Linear Models Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (3 votes)    </div>
    </div>
<div class="page" title="Page 1">
<div class="section">
<div class="section">
<div class="layoutArea">
<div class="column">
&nbsp;

</div>
</div>
<div class="layoutArea">
<div class="column">
Assignment 1

We explored linear models the last lecture. We will strengthen this understanding by implementing linear and logistic regression models as part of the assignment.

Section I – Linear Regression

We will implement a linear regression model to fit a curve to some data. Since the data is nonlinear, we will implement polynomial regression and use ridge regression to implement the best possible fit.

1. Load Data and Visualize

</div>
</div>
<div class="layoutArea">
<div class="column">
Let us load a dataset of points

</div>
<div class="column">
. As a first step, let’s import the required libraries followed by the dataset.

</div>
</div>
<div class="layoutArea">
<div class="column">
import numpy as np

from datasets import ridge_reg_data

<pre># Libraries for evaluating the solution
</pre>
import pytest

import numpy.testing as npt import random random.seed(1) np.random.seed(1)

train_X, train_Y, test_X, test_Y = ridge_reg_data() # Pre-defined function for loading the dataset train_Y = train_Y.reshape(-1,1) # reshaping from (m,) -&gt; (m,1)

test_Y = test_Y.reshape(-1,1)

print(‘train_X.shape is ‘, train_X.shape)

<pre>print('train_Y.shape is ', train_Y.shape)
print('test_X.shape is ', test_X.shape)
print('test_Y.shape is ', test_Y.shape)
</pre>
</div>
</div>
<div class="layoutArea">
<div class="column">
<pre>train_X.shape is  (300, 1)
train_Y.shape is  (300, 1)
test_X.shape is  (200, 1)
test_Y.shape is  (200, 1)
</pre>
Visualize Data

</div>
</div>
<div class="layoutArea">
<div class="column">
The dataset is split into train and test sets. The train set consists of 300 samples and the test set consists of 200 samples. We will use scatter plot to visualize the relationship between the ‘ ‘ and ‘ ‘. Lets visualize the data using the scatter plot from matplotlib.

</div>
</div>
<div class="layoutArea">
<div class="column">
In [4]:

In [5]:

Hint: You may want to use numpy.dot In [6]:

In [7]:

Hint: You may want to use numpy.linalg.inv and numpy.dot. In [8]:

In [9]:

In [117]:

In [118]:

In [119]:

In [120]:

In [121]:

In [122]:

In [123]:

In [124]:

In [125]:

In [126]:

In [127]:

In [128]:

In [717]:

In [718]:

In [719]:

In [720]:

In [721]:

In [722]:

In [723]:

In [724]:

In [725]:

In [726]:

In [727]:

In [728]:

In [729]:

In [730]:

In [731]:

In [732]:

In [733]:

In [734]:

In [661]:

<pre>  In [ ]:
  In [ ]:
  In [ ]:
  In [ ]:
</pre>
</div>
</div>
<div class="layoutArea">
<div class="column">
import matplotlib.pyplot as plt %matplotlib inline plt.scatter(train_X,train_Y,marker=’o’,s=4) plt.ylim(-2, 3)

<pre>plt.xlabel('x')
plt.ylabel('y');
</pre>
</div>
</div>
<div class="layoutArea">
<div class="column">
Linear Regression – Polynomial Transformation

Using the train data we hope to learn a relationship mapping to . We can evaluate this mapping using the test data. Linear regression will try to fit a straight line (linear relation) mapping to . However, we observe the and do not have a linear relationship. A straight line will not be a good fit. We need a non- linear mapping (curve) between and .

We discussed in the lecture that nonlinear regression can be achieved by transforming the scalar to a high dimension sample and performing linear regression with the transformed data. We can transform into a dimensional vector ( ) in order to perform nonlinear regression. For example,

</div>
</div>
<div class="layoutArea">
<div class="column">
transforms into a dimension vector

</div>
<div class="column">
, where is raised to . In vectorized notation, the dataset is transformed to is the number of samples.

</div>
</div>
<div class="layoutArea">
<div class="column">
of dimension

Every scalar is converted into a

</div>
<div class="column">
, where

</div>
</div>
<div class="layoutArea">
<div class="column">
dimension vector, In the above equation, is the target variable,

</div>
</div>
<div class="layoutArea">
<div class="column">
point in the row vector format, where is the

In the vectorized notation, the linear regression for dimensions ,

</div>
</div>
<div class="layoutArea">
<div class="column">
. We can now perform linear regression in dimensions.

are the parameters/weights of the model, is the transformed data

component.

samples is written as , where has the data points as row vectors and is of

</div>
</div>
<div class="layoutArea">
<div class="column">
, where is the number of samples and is the degree of the polynomial that we are trying to fit. The – Vector of the prediction labels of dimension . Lets implement a function to achieve this transformation.

</div>
</div>
<div class="layoutArea">
<div class="column">
– is the Design matrix of dimension

first column of 1’s in the design matrix will account for the bias , resulting in dimensions

</div>
</div>
<div class="layoutArea">
<div class="column">
def poly_transform(X,d): ”’

<pre>    Function to transform scalar values into (d+1)-dimension vectors.
    Each scalar value x is transformed a vector [1,x,x^2,x^3, ... x^d].
</pre>
<pre>    Inputs:
        X: vector of m scalar inputs od shape (m, 1) where each row is a scalar input x
        d: number of dimensions
</pre>
<pre>    Outputs:
        Phi: Transformed matrix of shape (m, (d+1))
</pre>
”’

Phi = np.ones((X.shape[0],1)) for i in range(1,d+1):

<pre>        col = np.power(X,i)
</pre>
Phi = np.hstack([Phi,col]) return Phi

</div>
</div>
<div class="layoutArea">
<div class="column">
Linear Regression – Objective Function (5 Points)

Let us define the objective function that will be optimized by the linear regression model.

Here, is the design matrix of dimensions (m \times (d+1)) and is the dimension vector of labels. is the dimension vector of weight parameters.

</div>
</div>
<div class="layoutArea">
<div class="column">
def lin_reg_obj(Y,Phi,theta): ”’

<pre>    Objective function to estimate loss for the linear regression model.
    Inputs:
</pre>
<pre>        Phi: Design matrix of dimensions (m, (d+1))
        Y: ground truth labels of dimensions (m, 1)
        theta: Parameters of linear regression of dimensions ((d+1),1)
</pre>
<pre>    outputs:
        loss: scalar loss
</pre>
”’

<pre>    # your code here
</pre>
prod = np.dot(Phi, theta)

loss = np.dot(np.transpose(Y-prod),(Y-prod)) return loss

</div>
</div>
<div class="layoutArea">
<div class="column">
<pre># Contains hidden tests
</pre>
<pre>random.seed(1)
np.random.seed(1)
m1 = 10;
d1 = 5;
</pre>
<pre>X_t = np.random.randn(m1,1)
Y_t = np.random.randn(m1,1)
theta_t = np.random.randn((d1+1),1)
PHI_t = poly_transform(X_t,d1)
loss_est = lin_reg_obj(Y_t,PHI_t,theta_t)
</pre>
</div>
</div>
<div class="layoutArea">
<div class="column">
Linear Regression – Closed Form Solution (10 Points)

Let us define a closed form solution to the objective function. Feel free to revisit the lecture to review the topic. Closed form solution is given by,

Here is the dimension design matrix obtained using poly_transform function defined earlier and are the ground truth labels of dimensions .

</div>
</div>
<div class="layoutArea">
<div class="column">
<pre>#Closed form solution
</pre>
def lin_reg_fit(Phi_X,Y): ”’

<pre>    A function to estimate the linear regression model parameters using the closed form solution.
    Inputs:
</pre>
<pre>        Phi_X: Design matrix of dimensions (m, (d+1))
        Y: ground truth labels of dimensions (m, 1)
</pre>
<pre>    Outputs:
        theta: Parameters of linear regression of dimensions ((d+1),1)
</pre>
”’

<pre>    # your code here
</pre>
<pre>    Phi_t = np.transpose(Phi_X)
    inverse_m = np.linalg.inv(np.dot(Phi_t, Phi_X))
    theta = np.dot(np.dot(inverse_m, Phi_t), Y)
</pre>
return theta

</div>
</div>
<div class="layoutArea">
<div class="column">
<pre># Contains hidden tests
</pre>
<pre>random.seed(1)
np.random.seed(1)
m1 = 10;
d1 = 5;
</pre>
<pre>X_t = np.random.randn(m1,1)
Y_t = np.random.randn(m1,1)
PHI_t = poly_transform(X_t,d1)
theta_est = lin_reg_fit(PHI_t,Y_t)
</pre>
</div>
</div>
<div class="layoutArea">
<div class="column">
Metrics for Evaluation (10 points)

We will evaluate the goodness of our linear regression model using root mean square error. This compares the difference between the estimate Y-labels and the groundth truth Y-labels. The smaller the RMSE value, better is the fit.

1. RMSE (Root Mean Squared Error)

Hint: You may want to use: numpy.sqrt, numpy.sum or numpy.dot.

Let’s visualize the nonlinear regression fit and the RMSE evaluation error on the test data

<pre>Train RMSE =  0.5136340617403364
Test RMSE =  0.5037691797614892
</pre>
2. Ridge Regression

The degree of the polynomial regression is . Even though the curve appears to be smooth, it may be fitting to the noise. We will use Ridge Regression to get a smoother fit and avoid-overfitting. Recall the ridge regression objective form:

where, is the regularization parameter. Larger the value of , the more smooth the curve. The closed form solution to the objective is give by:

Here, is the identity matrix of dimensions , is the dimension design matrix obtained using poly_transform function defined earlier and are the ground truth labels of dimensions .

Ridge Regression Closed Form Solution (5 points)

Similar to Linear regression, lets implement the closed form solution to ridge regression.

def ridge_reg_fit(Phi_X,Y,lamb_d): ”’

<pre>    A function to estimate the ridge regression model parameters using the closed form solution.
    Inputs:
</pre>
<pre>        Phi_X: Design matrix of dimensions (m, (d+1))
        Y: ground truth labels of dimensions (m, 1)
        lamb_d: regularization parameter
</pre>
<pre>    Outputs:
        theta: Parameters of linear regression of dimensions ((d+1),1)
</pre>
”’

#Step 1: get the dimension dplus1 using Phi_X to create the identity matrix $I_d$

#Step 2: Estimate the closed form solution similar to *linear_reg_fit* but now includethe lamb_d**2*I_d term # your code here

col_no = np.size(Phi_X, 1)

Phi_t = np.transpose(Phi_X)

Phi_product = np.dot(Phi_t, Phi_X)

identity_m = np.power(lamb_d, 2) * np.identity(col_no)

inverse_m = np.linalg.inv(Phi_product + identity_m)

theta = np.dot(np.dot(inverse_m, Phi_t), Y)

return theta

</div>
</div>
<div class="layoutArea">
<div class="column">
def get_rmse(Y_pred,Y): ”’

<pre>    function to evaluate the goodness of the linear regression model.
</pre>
<pre>    Inputs:
        Y_pred: estimated labels of dimensions (m, 1)
        Y: ground truth labels of dimensions (m, 1)
</pre>
<pre>    Outputs:
        rmse: root means square error
</pre>
”’

<pre>    # your code here
</pre>
<pre>    sum = 0
    m = np.size(Y,0)
</pre>
<pre>      for i in range(0, m-1):
         sum = sum + np.power(Y_pred[i] - Y[i], 2)
</pre>
diff = Y_pred – Y

diff_squared = diff ** 2 mean_diff = diff_squared.mean() rmse = np.sqrt(mean_diff) return rmse

</div>
</div>
<div class="layoutArea">
<div class="column">
# #

</div>
</div>
<div class="layoutArea">
<div class="column">
<pre># Contains hidden tests
</pre>
<pre>random.seed(1)
np.random.seed(1)
m1 = 50
Y_Pred_t = np.random.randn(m1,1)
Y_t = np.random.randn(m1,1)
rmse_est = get_rmse(Y_Pred_t,Y_t)
</pre>
</div>
</div>
<div class="layoutArea">
<div class="column">
d = 20

Phi_X_tr = poly_transform(train_X,d)

theta = lin_reg_fit(Phi_X_tr,train_Y) #Estimate the prediction on the train data Y_Pred_tr = np.dot(Phi_X_tr,theta)

rmse = get_rmse(Y_Pred_tr,train_Y) print(‘Train RMSE = ‘, rmse)

<pre>#Perform the same transform on the test data
</pre>
Phi_X_ts = poly_transform(test_X,d) #Estimate the prediction on the test data Y_Pred_ts = np.dot(Phi_X_ts,theta) #Evaluate the goodness of the fit

rmse = get_rmse(Y_Pred_ts,test_Y) print(‘Test RMSE = ‘, rmse)

import matplotlib.pyplot as plt

%matplotlib inline plt.scatter(test_X,test_Y,marker=’o’,s=4)

# Sampling more points to plot a smooth curve px = np.linspace(-2,2,100).reshape(-1,1)

<pre>PX = poly_transform(px,d)
py = np.dot(PX,theta)
plt.xlabel('x')
plt.ylabel('y')
plt.ylim(-2, 3)
plt.plot(px,py,color='red');
</pre>
</div>
</div>
<div class="layoutArea">
<div class="column">
<pre># Contains hidden tests
</pre>
<pre>random.seed(1)
np.random.seed(1)
m1 = 10;
d1 = 5;
</pre>
<pre>lamb_d_t = 0.1
X_t = np.random.randn(m1,1)
Y_t = np.random.randn(m1,1)
PHI_t = poly_transform(X_t,d1)
theta_est = ridge_reg_fit(PHI_t,Y_t,lamb_d_t)
</pre>
Cross Validation to Estimate ( )

In order to avoid overfitting when using a high degree polynomial, we have used ridge regression. We now need to estimate the optimal value of

cross-validation.

</div>
<div class="column">
using

</div>
</div>
<div class="layoutArea">
<div class="column">
We will obtain a generic value of using the entire training dataset to validate. We will employ the method of -fold cross validation, where we split the training data into non-overlapping random subsets. In every cycle, for a given value of , subsets are used for training the ridge regression model and the remaining subset is used for evaluating the goodness of the fit. We estimate the average goodness of the fit across all the subsets and select the

that results in the best fit. K-fold cross validation

It is easier to shuffle the index and slice the training into required number of segments, than processing the complete dataset. The below function k_val_ind returns a 2D list of indices by spliting the datapoints into ‘ ‘ sets

Refer the following documentation for splitting and shuffling:

https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.shuffle.html https://docs.scipy.org/doc/numpy/reference/generated/numpy.split.html

def k_val_ind(index,k_fold,seed=1): ”’

<pre>    Function to split the data into k folds for cross validation. Returns the indices of the data points
    belonging to every split.
</pre>
<pre>    Inputs:
        index: all the indices of the training
        k_fold: number of folds to split the data into
</pre>
<pre>    Outputs:
        k_set: list of arrays with indices
</pre>
”’

np.random.seed(seed)

np.random.shuffle(index) # Shuffle the indices

k_set = np.split(index,k_fold) # Split the indices into ‘k_fold’ return k_set

K- Fold Cross Validation (10 Points)

Let’s now implement -fold cross validation.

def k_fold_cv(k_fold,train_X,train_Y,lamb_d,d): ”’

<pre>    Function to implement k-fold cross validation.
    Inputs:
</pre>
<pre>        k_fold: number of validation subsests
        train_X: training data of dimensions (m, 1)
        train_Y: ground truth training labels
        lamb_d: ridge regularization lambda parameter
        d: polynomial degree
</pre>
<pre>    Outputs:
        rmse_list: list of root mean square errors (RMSE) for k_folds
</pre>
”’

index = np.arange(train_X.shape[0]) # indices of the training data

k_set = k_val_ind(index,k_fold) # pre-defined function to shuffle and split indices

Phi_X = poly_transform(train_X, d) #transform all the data to (m,(d+1)) dimensions rmse_list = []

for i in range(k_fold):

ind = np.zeros(train_X.shape[0], dtype=bool) # binary mask ind[k_set[i]] = True # validation portion is indicated

<pre>        #Note: Eg. train_X[ind] -&gt; validation set, train_X[~ind] -&gt; training set
        # Write your answer inside the 'for' loop
        # Note: Phi_X[~ind,:] is training subset and Phi_X[ind,:] is validation subset. Similary for the train and v
</pre>
<pre>alidation labels.
        # Step 1: Estimate the theta parameter using ridge_reg_fit with the training subset, training labels and lam
</pre>
</div>
</div>
<div class="layoutArea">
<div class="column">
b_d

</div>
</div>
<div class="layoutArea">
<div class="column">
<pre># Step 2: Estimate the prediction Y_pred over the validation as a dot product over Phi_X[ind,:] and theta
# Step 3: use 'get_rmse' function to determine rmse using Y_pred and train_Y[ind]
</pre>
<pre># your code here
</pre>
<pre>theta = ridge_reg_fit(Phi_X[~ind,:], train_Y[~ind,:], lamb_d)
Y_pred = np.dot(Phi_X[ind,:],theta)
rmse = get_rmse(Y_pred, train_Y[ind])
</pre>
</div>
</div>
<div class="layoutArea">
<div class="column">
rmse_list.append(rmse) return rmse_list

</div>
</div>
<div class="layoutArea">
<div class="column">
<pre># Contains hidden tests
</pre>
np.random.seed(1)

m1 = 20;

d1 = 5;

k_fold_t = 5 # number of portions to split the training data lamb_d_t = 0.1

<pre>X_t = np.random.randn(m1,1)
Y_t = np.random.randn(m1,1)
</pre>
rmse_list_est = k_fold_cv(k_fold_t,X_t,Y_t,lamb_d_t,d1)

Let us select the value of that provides the lowest error based on RMSE returned by the ‘k_fold_cv’ function.

In this example, we will choose the best value of among 6 values.

k_fold = 5

l_range = [0,1e-3,1e-2,1e-1,1,10] # The set of lamb_d parameters used for validation. th = float(‘inf’)

for lamb_d in l_range:

print(‘lambda:’+str(lamb_d))

rmse = k_fold_cv(k_fold,train_X,train_Y,lamb_d,d) print(“RMSE: “,rmse)

print(“*************”)

mean_rmse = np.mean(rmse)

if mean_rmse&lt;th:

<pre>        th = mean_rmse
        l_best = lamb_d
</pre>
<pre>print("Best value for the regularization parameter(lamb_d):",l_best)
</pre>
<pre>lambda:0
RMSE:  [0.900555177526016, 0.5995063480546855, 0.48899370047004265, 0.5734994260228984, 0.5778294698629982]
*************
lambda:0.001
RMSE:  [0.9254777112353597, 0.6018895272984746, 0.4886770424932113, 0.5708466724857937, 0.5784618729196633]
*************
lambda:0.01
RMSE:  [1.0459044884476891, 0.625182224075695, 0.493313775686007, 0.5570647419146361, 0.5899621997377145]
*************
lambda:0.1
RMSE:  [0.826147452070392, 0.646524587121386, 0.4903308187338262, 0.5660950349966889, 0.5945668688216352]
*************
lambda:1
RMSE:  [0.6799665144970705, 0.6886693542491483, 0.5647357620788945, 0.6393074933827457, 0.6470329335868142]
*************
lambda:10
RMSE:  [0.7335261033175889, 0.6993069184592259, 0.7556132543797057, 0.7992608698443789, 0.8199075014141747]
*************
Best value for the regularization parameter(lamb_d): 0.1
</pre>
Evaluation on the Test Set (10 Points)

As discussed in previous section, we will present the final evaluation of the model based on the test set.

<pre>lamb_d = l_best
</pre>
<pre># Step 1: Create Phi_X using 'poly_transform(.)' on the train_X and d=20
# Step 2: Estimate theta using ridge_reg_fit(.) with Phi_X, train_Y and the best lambda
# Step 3: Create Phi_X_test using 'poly_transform(.)' on the test_X and d=20
# Step 4: Estimate the Y_Pred for the test data using Phi_X_test and theta
# Step 5: Estimate rmse using get_rmse(.) on the Y_Pred and test_Y
</pre>
</div>
</div>
<div class="layoutArea">
<div class="column">
<pre># your code here
</pre>
<pre>d = 20
Phi_X = poly_transform(train_X, d)
theta = ridge_reg_fit(Phi_X, train_Y, lamb_d)
Phi_X_test = poly_transform(test_X, d)
Y_pred = np.dot(Phi_X_test, theta)
rmse = get_rmse(Y_pred, test_Y)
</pre>
<pre>print("RMSE on test set is "+ str(rmse))
RMSE on test set is 0.49850101448090123
</pre>
<pre># Contains hidden tests checking for rmse &lt; 0.5
</pre>
Let’s visualize the model’s prediction on the test data set.

<pre>print('Test RMSE = ', rmse)
</pre>
%matplotlib inline plt.scatter(test_X,test_Y,marker=’o’,s=4)

# Sampling more points to plot a smooth curve px = np.linspace(-2,2,100).reshape(-1,1)

PX = poly_transform(px,d)

py = np.dot(PX,theta)

plt.xlabel(‘x’)

plt.ylabel(‘y’)

plt.ylim(-2, 3)

plt.plot(px,py,color=’red’);

<pre>Test RMSE =  0.49850101448090123
</pre>
You have completed linear ridge regression and estimated the best value for the regularization parameter

Section II – Logistic Regression

</div>
<div class="column">
using k-fold cross validation.

</div>
</div>
<div class="layoutArea">
<div class="column">
Machine learning is used in medicine for assisting doctors with crucial decision-making based on dignostic data. In this assignment we will be designing a logistic regression model (single layer neural network) to predict if a subject is diabetic or not. The model will classify the subjects into two groups diabetic (Class 1) or non-diabetic (Class 0) – a binary classification model.

We will be using the ‘Pima Indians Diabetes dataset’ to train our model which contains different clinical parameters (features) for multiple subjects along with the label (diabetic or not-diabetic). Each subject is represented by 8 features (Pregnancies, Glucose, Blood-Pressure, SkinThickness, Insulin, BMI, Diabetes- Pedigree-Function, Age) and the ‘Outcome’ which is the class label. The dataset contains the results from 768 subjects.

</div>
</div>
<div class="layoutArea">
<div class="column">
We will be spliting the dataset into train and test data. We will train our model on the train data and predict the categories on the test data.

<pre>#importing a few libraries
</pre>
import numpy as np

from datasets import pima_data import sys

import matplotlib.pyplot as plt import numpy.testing as npt

1. Load Data, Visualize and Normalize

Let us load the training and test data.

<pre>train_X,train_Y,test_X,test_Y  = pima_data()
</pre>
<pre>print('train_X.shape = ', train_X.shape)
print('train_Y.shape = ', train_Y.shape)
print('test_X.shape = ', test_X.shape)
print('test_Y.shape = ', test_Y.shape)
</pre>
<pre># Lets examine the data
</pre>
print(‘\nFew Train data examples’) print(train_X[:5, :])

print(‘\nFew Train data labels’) print(train_Y[:5])

<pre>train_X.shape =  (500, 8)
train_Y.shape =  (500,)
test_X.shape =  (268, 8)
test_Y.shape =  (268,)
</pre>
<pre>Few Train data examples
[[6.000e+00 1.480e+02 7.200e+01 3.500e+01 0.000e+00 3.360e+01 6.270e-01
</pre>
<pre>  5.000e+01]
 [1.000e+00 8.500e+01 6.600e+01 2.900e+01 0.000e+00 2.660e+01 3.510e-01
</pre>
<pre>  3.100e+01]
 [8.000e+00 1.830e+02 6.400e+01 0.000e+00 0.000e+00 2.330e+01 6.720e-01
</pre>
<pre>  3.200e+01]
 [1.000e+00 8.900e+01 6.600e+01 2.300e+01 9.400e+01 2.810e+01 1.670e-01
</pre>
<pre>  2.100e+01]
 [0.000e+00 1.370e+02 4.000e+01 3.500e+01 1.680e+02 4.310e+01 2.288e+00
</pre>
3.300e+01]]

<pre>Few Train data labels
[1. 0. 1. 0. 1.]
</pre>
# We notice the data is not normalized. Lets do a simple normalization scaling to data between 0 and 1 # Normalized data is easier to train using large learning rates

train_X = np.nan_to_num(train_X/train_X.max(axis=0))

test_X = np.nan_to_num(test_X/test_X.max(axis=0))

#Lets reshape the data so it matches our notation from the lecture.

#train_X should be (d, m) and train_Y should (1,m) similarly for test_X and test_Y train_X = train_X.T

train_Y= train_Y.reshape(1,-1)

<pre>test_X = test_X.T
test_Y= test_Y.reshape(1,-1)
print('train_X.shape = ', train_X.shape)
print('train_Y.shape = ', train_Y.shape)
print('test_X.shape = ', test_X.shape)
print('test_Y.shape = ', test_Y.shape)
</pre>
<pre># Lets examine the data and verify it is normalized
</pre>
print(‘\nFew Train data examples’) print(train_X[:, :5])

print(‘\nFew Train data labels’) print(train_Y[0,:5])

<pre>train_X.shape =  (8, 500)
train_Y.shape =  (1, 500)
test_X.shape =  (8, 268)
test_Y.shape =  (1, 268)
</pre>
<pre>Few Train data examples
[[0.35294118 0.05882353 0.47058824 0.05882353 0.        ]
</pre>
<pre> [0.74371859 0.42713568 0.91959799 0.44723618 0.68844221]
 [0.59016393 0.54098361 0.52459016 0.54098361 0.32786885]
 [0.35353535 0.29292929 0.         0.23232323 0.35353535]
 [0.         0.         0.         0.11111111 0.19858156]
 [0.50074516 0.39642325 0.34724292 0.41877794 0.64232489]
 [0.25909091 0.14504132 0.27768595 0.06900826 0.94545455]
 [0.61728395 0.38271605 0.39506173 0.25925926 0.40740741]]
</pre>
<pre>Few Train data labels
[1. 0. 1. 0. 1.]
</pre>
<pre>#There are 8 features for each of the data points. Lets plot the data using a couple of features
</pre>
fig, ax = plt.subplots()

plt.scatter(train_X[6,:],train_X[7,:], c=train_Y[0]) plt.xlabel(‘Diabetes-Pedigree-Function’)

plt.ylabel(‘Age’)

plt.show();

# We have plotted train_X[6,:],train_X[7,:].

# Feel free to insert your own cells to plot and visualize different variable pairs.

2. Quick Review of the Steps Involved in Logistic Regression Using Gradient Descent.

</div>
</div>
<div class="layoutArea">
<div class="column">
1. Training data is of dimensions where is number of features and

</div>
<div class="column">
is number of samples. Training labels is of dimensions

</div>
<div class="column">
.

is

</div>
</div>
<div class="layoutArea">
<div class="column">
2. Initilaize logistic regression model parameters and set to zero

3. Calculate using and intial parameter values 4. Apply the sigmoid activation to estimate on ,

5. Calculate the loss between predicted probabilities

6. Calculate gradient dZ (or ),

7. Calculate gradients represented by , represented by

8. Adjust the model parameters using the gradients. Here is the learning rate.

</div>
</div>
<div class="layoutArea">
<div class="column">
where is of dimensions and

</div>
<div class="column">
is a scalar.

</div>
<div class="column">
is initialized to small random values and

</div>
</div>
<div class="layoutArea">
<div class="column">
and groundtruth labels ,

</div>
</div>
<div class="layoutArea">
<div class="column">
9. Loop until the loss converges or for a fixed number of epochs. We will first define the functions logistic_loss() and grad_fn() along with other functions below.

Review

Lecture Notes

Intialize Parameters (5 Points)

we will initialize the model parameters. The weights will be initialized with small random values and bias as 0. While the bias will be a scalar, the dimension of weight vector will be , where is the number of features.

Hint:np.random.randn can be used here to create a vector of random integers of desired shape.

def initialize(d, seed=1): ”’

<pre>    Function to initialize the parameters for the logisitic regression model
</pre>
<pre>    Inputs:
        d: number of features for every data point
        seed: random generator seed for reproducing the results
</pre>
<pre>    Outputs:
        w: weight vector of dimensions (d, 1)
        b: scalar bias value
</pre>
”’

<pre>    np.random.seed(seed)
</pre>
# NOTE: initialize w to be a (d,1) column vector instead of (d,) vector

# Hint: initialize w to a random vector with small values. For example, 0.01*np.random.randn(.) can be used. # and initialize b to scalar 0

# your code here

w = 0.01 * np.random.randn(d, 1)

b=0

return w,b

<pre># Contains hidden tests
</pre>
Sigmoid Function (5 Points)

Let’s now implement Sigmoid activation function.

where z is in the input variable. Hint: numpy.exp can be used for defining the exponential function.

def sigmoid(z):

# your code here

A = 1./(1 + np.exp(-1*z)) return A

<pre># Contains hidden tests
</pre>
np.random.seed(1)

d=2

m1 = 5

X_t = np.random.randn(d,m1)

Logistic Loss Function (5 Points)

We will define the objective function that will be used later for determining the loss between the model prediction and groundtruth labels. We will use vectors (activation output of the logistic neuron) and (groundtruth labels) for defining the loss.

where is the number of input datapoints and is used for averaging the total loss. Hint: numpy.sum and numpy.log.

def logistic_loss(A,Y): ”’

<pre>    Function to calculate the logistic loss given the predictions and the targets.
</pre>
<pre>    Inputs:
        A: Estimated prediction values, A is of dimension (1, m)
        Y: groundtruth labels, Y is of dimension (1, m)
</pre>
<pre>    Outputs:
        loss: logistic loss
</pre>
”’

m = A.shape[1]

# your code here

calc = np.dot(np.log(A), np.transpose(Y)) + np.dot(np.log(1 – A), np.transpose(1-Y)) sum = np.sum(calc)

loss = (-1/m) * sum

return loss

</div>
</div>
<div class="layoutArea">
<div class="column">
<pre># Contains hidden tests
</pre>
np.random.seed(1)

d=2

m1 = 10

X_t = np.random.randn(d,m1) Y_t = np.random.rand(1,m1) Y_t[Y_t&gt;0.5] = 1 Y_t[Y_t&lt;=0.5] = 0

Gradient Function (5 Points)

Let us define the gradient function for calculating the gradients ( The gradients can be calculated as,

</div>
<div class="column">
). We will use it during gradient descent.

</div>
</div>
<div class="layoutArea">
<div class="column">
Instead of , we will use dZ (or ) since,

Make sure the gradients are of correct dimensions. Refer to lecture for more information.

Hint: numpy.dot and numpy.sum. Check use of ‘keepdims’ parameter.

def grad_fn(X,dZ): ”’

<pre>    Function to calculate the gradients of weights (dw) and biases (db) w.r.t the objective function L.
</pre>
<pre>    Inputs:
        X: training data of dimensions (d, m)
        dZ: gradient dL/dZ where L is the logistic loss and Z = w^T*X+b is the input to the sigmoid activation funct
</pre>
</div>
</div>
<div class="layoutArea">
<div class="column">
ion

”’

m = X.shape[1]

# your code here

dw = 1/m * np.dot(X, np.transpose(dZ)) db = 1/m * np.sum(dZ)

return dw,db

</div>
</div>
<div class="layoutArea">
<div class="column">
<pre>dZ is of dimensions (1, m)
</pre>
</div>
</div>
<div class="layoutArea">
<div class="column">
<pre>outputs:
    dw: gradient dL/dw - gradient of the weight w.r.t. the logistic loss. It is of dimensions (d,1)
    db: gradient dL/db - gradient of the bias w.r.t. the logistic loss. It is a scalar
</pre>
</div>
</div>
<div class="layoutArea">
<div class="column">
<pre># Contains hidden tests
</pre>
np.random.seed(1)

d=2

m1 = 10

X_t = np.random.randn(d,m1) Y_t = np.random.rand(1,m1) Y_t[Y_t&gt;0.5] = 1 Y_t[Y_t&lt;=0.5] = 0

Training the Model (10 Points)

We will now implement the steps for gradient descent discussed earlier.

def model_fit(w,b,X,Y,alpha,n_epochs,log=False): ”’

<pre>    Function to fit a logistic model with the parameters w,b to the training data with labels X and Y.
</pre>
<pre>    Inputs:
        w: weight vector of dimensions (d, 1)
        b: scalar bias value
        X: training data of dimensions (d, m)
        Y: training data labels of dimensions (1, m)
        alpha: learning rate
        n_epochs: number of epochs to train the model
</pre>
<pre>    Outputs:
        params: a dictionary to hold parameters w and b
        losses: a list train loss at every epoch
</pre>
”’

losses=[]

for epoch in range(n_epochs):

<pre>        # Implement the steps in the logistic regression using the functions defined earlier.
        # For each iteration of the for loop
</pre>
<pre>            # Step 1: Calculate output Z = w.T*X + b
            # Step 2: Apply sigmoid activation: A = sigmoid(Z)
            # Step 3: Calculate loss = logistic_loss(.) between predicted values A and groundtruth labels Y
            # Step 4: Estimate gradient dZ = A-Y
            # Step 5: Estimate gradients dw and db using grad_fn(.).
            # Step 6: Update parameters w and b using gradients dw, db and learning rate
</pre>
<ul>
<li>
<pre>            # &nbsp;        w = w - alpha * dw
</pre>
</li>
<li>
<pre>            # &nbsp;        b = b - alpha * db
</pre>
<pre>        # your code here
</pre>
<pre>        Z = np.dot(np.transpose(w), X) + b
        A = sigmoid(Z)
        loss = logistic_loss(A, Y)
        dZ = A-Y
</pre>
dw,db = grad_fn(X,dZ) w = w – alpha * dw

b = b – alpha * db

if epoch%100 == 0:

losses.append(loss) if log == True:

print(“After %i iterations, Loss = %f”%(epoch,loss)) params ={“w”:w,”b”:b}
</li>
</ul>
</div>
</div>
<div class="layoutArea">
<div class="column">
return params,losses # Contains hidden tests

np.random.seed(1)

d=2

m1 = 10

X_t = np.random.randn(d,m1) Y_t = np.random.rand(1,m1) Y_t[Y_t&gt;0.5] = 1 Y_t[Y_t&lt;=0.5] = 0

Model Prediction (10 Points)

Once we have the optimal values of model parameters

</div>
<div class="column">
, we can determine the accuracy of the model on the test data.

</div>
</div>
<div class="layoutArea">
<div class="column">
def model_predict(params,X,Y=np. array([]),pred_threshold=0.5): ”’

<pre>    Function to calculate category predictions on given data and returns the accuracy of the predictions.
    Inputs:
</pre>
<pre>        params: a dictionary to hold parameters w and b
        X: training data of dimensions (d, m)
        Y: training data labels of dimensions (1, m). If not provided, the function merely makes predictions on X
</pre>
<pre>    outputs:
        Y_Pred: Predicted class labels for X. Has dimensions (1, m)
        acc: accuracy of prediction over X if Y is provided else, 0
        loss: loss of prediction over X if Y is provided else, Inf
</pre>
”’

<pre>    w = params['w']
    b = params['b']
    m = X.shape[1]
</pre>
# Calculate Z using X, w and b

# Calculate A using the sigmoid – A is the set of (1,m) probabilities

# Calculate the prediction labels Y_Pred of size (1,m) using A and pred_threshold # When A&gt;pred_threshold Y_Pred is 1 else 0

# your code here

Z = np.dot(np.transpose(w), X) + b

A = sigmoid(Z)

Y_Pred = np.copy(A)

for i in range(len(A[0,:])): if A[0][i]&gt;pred_threshold:

Y_Pred[0][i] = 1 else:

<pre>             Y_Pred[0][i] = 0
    Y_Pred = Y_Pred.reshape(1, -1)
</pre>
acc = 0

loss = float(‘inf’) if Y.size!=0:

<pre>        loss = logistic_loss(A,Y)
</pre>
acc = np.mean(Y_Pred==Y) return Y_Pred, acc, loss

<pre># Contains hidden tests
</pre>
np.random.seed(1) d=2

m1 = 10

<pre># Test standard
</pre>
<pre>X_t = np.random.randn(d,m1)
Y_t = np.random.rand(1,m1)
Y_t[Y_t&gt;0.5] = 1
Y_t[Y_t&lt;=0.5] = 0
</pre>
3. Putting it All Together (10 Points)

We will train our logistic regression model using the data we have loaded and test our predictions on diabetes classification.

<pre>#We can use a decently large learning rate becasue the features have been normalized
#When features are not normalized, larger learning rates may cause the learning to oscillate
#and go out of bounds leading to 'nan' errors
#Feel free to adjust the learning rate alpha and the n_epochs to vary the test accuracy
#You should be able to get test accuracy &gt; 70%
#You can go up to 75% to 80% test accuracies as well
</pre>
<pre>alpha = 0.15
n_epochs = 8000
</pre>
<pre># Write code to initialize parameters w and b with initialize(.) (use train_X to get feature dimensions d)
# Use model_fit(.) to estimate the updated 'params' of the logistic regression model and calculate how the 'losses'
</pre>
varies

# Use variables ‘params’ and ‘losses’ to store the outputs of model_fit(.) # your code here

w,b = initialize(np.size(train_X,0), seed=1)

params,losses = model_fit(w,b,train_X,train_Y,alpha,n_epochs,log=False)

<pre>Y_Pred_tr, acc_tr, loss_tr = model_predict(params,train_X,train_Y)
Y_Pred_ts, acc_ts, loss_ts = model_predict(params,test_X,test_Y)
print("Train Accuracy of the model:",acc_tr)
print("Test Accuracy of the model:",acc_ts)
</pre>
import matplotlib.pyplot as plt %matplotlib inline plt.plot(losses) plt.xlabel(‘Iterations(x100)’) plt.ylabel(‘Train loss’);

<pre>Train Accuracy of the model: 0.782
Test Accuracy of the model: 0.7201492537313433
</pre>
<pre># Contains hidden tests testing accuracy of test to be greater than 0.7 with the above parameter settings
</pre>
Congratulations on completing this week’s assignment – building a single leayer neural network for binary classification. In the following weeks, we will learn to build and train a multilayer neural network for multi category classification.

</div>
</div>
</div>
</div>
<div class="section">
<div class="layoutArea">
<div class="column">
A

</div>
</div>
<div class="layoutArea">
<div class="column">
)(

</div>
</div>
<div class="layoutArea">
<div class="column">
b )m×1(

</div>
<div class="column">
Y

</div>
<div class="column">
m

</div>
<div class="column">
)b,w(

d )m×d(

</div>
<div class="column">
X

</div>
</div>
<div class="layoutArea">
<div class="column">
5=d

</div>
</div>
<div class="layoutArea">
<div class="column">
λ

</div>
</div>
<div class="layoutArea">
<div class="column">
]dx,..,1x,1[ = )x(Φ )1 + d(

</div>
<div class="column">
htk

</div>
<div class="column">
kx

y

</div>
</div>
<div class="layoutArea">
<div class="column">
X

</div>
<div class="column">
k

</div>
<div class="column">
x kx ⊤]5x,4x,3x,2x,x,1[ 2≥d dx

</div>
</div>
<div class="layoutArea">
<div class="column">
Y

</div>
<div class="column">
)1×m( ))1+d(×m( )X(Φ

</div>
</div>
<div class="layoutArea">
<div class="column">
)1+d( θ

</div>
<div class="column">
m Y

)θ)X(Φ− Y(⊤)θ)X(Φ− Y(=)θ,Y,)X(Φ(L

</div>
<div class="column">
)X(Φ

</div>
</div>
<div class="layoutArea">
<div class="column">
λ

</div>
</div>
<div class="layoutArea">
<div class="column">
x

</div>
</div>
<div class="layoutArea">
<div class="column">
)Z(σ=A

b + X Tw = Z

</div>
<div class="column">
)b ,w(

</div>
</div>
<div class="layoutArea">
<div class="column">
) Y − A( = Zd

</div>
</div>
<div class="layoutArea">
<div class="column">
1=i m ))i(y−)i(a(∑ 1 =bd

</div>
</div>
<div class="layoutArea">
<div class="column">
m T)Y−A(Xm =wd

</div>
</div>
<div class="layoutArea">
<div class="column">
1=i m ))i(a−1(gol))i(y−1(+)i(agol)i(y∑ 1 −=)Y,A(L

</div>
</div>
<div class="layoutArea">
<div class="column">
)z−(pxe+1 =)z(σ 1

</div>
</div>
<div class="layoutArea">
<div class="column">
bd.α−b=:b wd.α−w=:w

</div>
</div>
<div class="layoutArea">
<div class="column">
α )Zd ,X(nf_darg = bd ,wd

</div>
</div>
<div class="layoutArea">
<div class="column">
) Y − A( = Zd

</div>
</div>
<div class="layoutArea">
<div class="column">
1

bd ,wd

</div>
</div>
<div class="layoutArea">
<div class="column">
)1−k( λ kλ

</div>
<div class="column">
k

</div>
</div>
<div class="layoutArea">
<div class="column">
Ld Ld

</div>
</div>
<div class="layoutArea">
<div class="column">
bd

</div>
<div class="column">
bd wd Ld

</div>
<div class="column">
wd Ld

</div>
</div>
<div class="layoutArea">
<div class="column">
m

</div>
<div class="column">
Y

</div>
</div>
<div class="layoutArea">
<div class="column">
) Y ,A(ssol_citsigol = ssol

Y A )(L

</div>
</div>
<div class="layoutArea">
<div class="column">
)Z−(pxe+1 =A 1

</div>
<div class="column">
ZA

</div>
</div>
<div class="layoutArea">
<div class="column">
b + X⊤w = Z wb)1,d( wbw

</div>
<div class="column">
X

</div>
<div class="column">
Z

</div>
</div>
<div class="layoutArea">
<div class="column">
dlo f_k

</div>
</div>
<div class="layoutArea">
<div class="column">
)1×m( Y ))1+d(×m( )X(Φ ))1+d(×)1+d((

</div>
</div>
<div class="layoutArea">
<div class="column">
Y⊤)X(Φ1−)dI2λ + )X(Φ⊤)X(Φ( = θ θ⊤θ2λ+)θ)X(Φ− Y(⊤)θ)X(Φ− Y(=)λ,θ,Y,)X(Φ(L

</div>
</div>
<div class="layoutArea">
<div class="column">
1=i m√ 2))i(y − )i(derp_y(∑ 1

</div>
</div>
<div class="layoutArea">
<div class="column">
−−−−−−−−−−−−−−−m−−−−

</div>
</div>
<div class="layoutArea">
<div class="column">
Y⊤)X(Φ1−))X(Φ⊤)X(Φ( = θ

</div>
</div>
<div class="layoutArea">
<div class="column">
d

d ⎦ dx …

</div>
<div class="column">
1+d

2x 1x 1⎣

</div>
<div class="column">
m )1+d(×m ^y

</div>
<div class="column">
X

</div>
</div>
<div class="layoutArea">
<div class="column">
⎦θ⎣ )m(

</div>
</div>
<div class="layoutArea">
<div class="column">
⎦)m( ⎣ ⎥1θ⎢⎥ dx … 2x 1x 1⎢ ⎥ ^y⎢

</div>
</div>
<div class="layoutArea">
<div class="column">
)m( )m(

</div>
</div>
<div class="layoutArea">
<div class="column">
⎥⋮⎢⎥⋮ ⋮ ⋮ ⋮ ⋮⎢⎥⋮⎢

</div>
</div>
<div class="layoutArea">
<div class="column">
⎥ ⎢⎥ ⎢=⎥ ⎢

</div>
</div>
<div class="layoutArea">
<div class="column">
⎤

</div>
<div class="column">
⎡⎥ )2( )2( )2( ⎢ )2(

</div>
</div>
<div class="layoutArea">
<div class="column">
0θ ⎤ dx … 2x 1x 1⎡ ⎤)1(^y⎡ )1( )1( )1(

</div>
<div class="column">
)1+d(×m

</div>
</div>
<div class="layoutArea">
<div class="column">
)X(Φ θ)X(Φ=^Y

</div>
<div class="column">
m ⊤]dθ,..,0θ[ = θ

</div>
</div>
<div class="layoutArea">
<div class="column">
dθdx + 1−dθ1−dx+…+1θ1x + 0θ = θ)x(Φ = y ⊤]dx,…,3x,2x,1x,1[

</div>
<div class="column">
x )1+d( x

</div>
</div>
<div class="layoutArea">
<div class="column">
λ

</div>
</div>
<div class="layoutArea">
<div class="column">
yx yx yx

</div>
</div>
<div class="layoutArea">
<div class="column">
yx

</div>
</div>
<div class="layoutArea">
<div class="column">
Zd Ld

</div>
<div class="column">
)Y−A(

</div>
</div>
<div class="layoutArea">
<div class="column">
λ

</div>
</div>
<div class="layoutArea">
<div class="column">
d )1×d(

</div>
</div>
<div class="layoutArea">
<div class="column">
dI λ 0≥λ

</div>
</div>
<div class="layoutArea">
<div class="column">
01 = d

</div>
</div>
<div class="layoutArea">
<div class="column">
Zd Ld

</div>
</div>
<div class="layoutArea">
<div class="column">
1×m Y

</div>
</div>
<div class="layoutArea">
<div class="column">
λ

</div>
</div>
<div class="layoutArea">
<div class="column">
)1 + d(

m )1+d(×m )X(Φ

</div>
</div>
<div class="layoutArea">
<div class="column">
yx

</div>
</div>
<div class="layoutArea">
<div class="column">
)y ,x(

</div>
</div>
<div class="layoutArea">
<div class="column">
k

</div>
</div>
<div class="layoutArea">
<div class="column">
m

</div>
</div>
<div class="layoutArea">
<div class="column">
adbmal

</div>
</div>
</div>
</div>
