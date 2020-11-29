import pandas as pd
import matplotlib.pyplot as plt

df= pd.read_csv('E:\\Projects\\GradientDescent\\Gradient Descent\\student_scores.csv')


def plot_regression_line(X,m,b):
    regression_y = []
    for x in X.values:
        y1=m*x+b
        regression_y.append(y1)
        
    return regression_y
    


def gradient_descent(all_x,all_y,m,b):
    totalError = 0
    for x, y_actual in zip(all_x,all_y):
        y_pred = m*x+b
        
        error = y_pred-y_actual
        totalError += error
        delta_m = error *x
        delta_b = error
        
        m= m- delta_m*0.001
        b= b- delta_b*0.001
        
    return m, b, totalError
    


x= df['Hours']
y = df['Scores']
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.title('Hours vs Scores')
plt.plot(x,y,'o')
 
m=0;b=0
for i in range(1,10):
    m, b, totalError= gradient_descent(x,y, m, b)
    regression_line = plot_regression_line(x,m,b)
    plt.plot(x,regression_line)
plt.show()
