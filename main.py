import helpers
from logistic_regression import sigmoid

if __name__ == '__main__':
    data = helpers.create_data()  # generate synthetic datasets
    sigmoid_obj = sigmoid(Data_points=data)  # creating a sigmoid object

    y_hat = sigmoid_obj.calculate_y_yat()  # map each Xi into [0,1]
    sigmoid_obj.plot_graph(y_hat)  # plot the output with (X,y_hat) for each (Xi,y_hat_i)

    # loss_0 = helpers.get_loss_y_0(y_hat)
    # loss_1 = helpers.get_loss_y_1(y_hat)

    loss_0 = helpers.get_loss_gen(y=0,y_hat=y_hat)
    loss_1 = helpers.get_loss_gen(y=1,y_hat=y_hat)

    helpers.plot_losses(y_hat=y_hat, loss_y_0=loss_0, loss_y_1=loss_1)
