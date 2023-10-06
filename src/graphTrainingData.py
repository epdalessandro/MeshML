import matplotlib.pyplot as plt

def getLists(line: str) -> "list[float]":
    startIdx = line.find("[") + 1
    line = line[startIdx:-2] # TODO: Why is this -2?
    line = line.replace(",", "")
    return [float(val) for val in line.split()] 

def getMetric(line: str) -> float:
    startIdx = line.find(":")+1
    num = line[startIdx:]
    return float(num)

def plotMetric(metric: "list[float]", labels: "list[str]", metricName: str, identifier: str):
    print("Plotting " + metricName)
    plt.figure()
    for row in range(len(labels)):
        plt.plot(metric[row], label=labels[row])
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel(metricName)
    plt.savefig(metricName + identifier + ".png")
    plt.close()

def plotEvals(eval, sizes, evalName: str):
    plt.figure()
    for idx in range(len(sizes)):
        plt.scatter([sizes[idx]]*5, eval[idx][:], label = str(sizes[idx]))
    plt.legend()
    plt.xlabel("BatchSize")
    plt.ylabel(evalName)
    plt.savefig(evalName + ".png")
    plt.close()

def main():
    modelNames = ["normal", "paper", "biggerH1H2", "biggerH3", "biggerHidden"]
    batchSizes = ["256", "512", "1024", "2048", "4096", "8192", "16384"]

    # Compare models
    mean_list, mse_list, mae_list, mape_list = [],[],[],[]
    for size in batchSizes:
        loss_list, valLoss_list, MSE_list, MAPE_list, labels = [], [], [], [], []
        eval_mean, eval_mse, eval_mae, eval_mape = [], [], [], []
        for name in modelNames:
            labels.append(name + "_" + size)
            fileName = "./output/" + name + size + "_data.csv"
            with open(fileName, "r") as file:
                file.readline() # Training metrics header
                loss_list.append(getLists(file.readline()))
                valLoss_list.append(getLists(file.readline()))
                MSE_list.append(getLists(file.readline()))
                MAPE_list.append(getLists(file.readline()))
                file.readline() # Evaluation metrics header
                eval_mean.append(getMetric(file.readline()))
                eval_mse.append(getMetric(file.readline()))
                eval_mae.append(getMetric(file.readline()))
                eval_mape.append(getMetric(file.readline()))
        mean_list.append(eval_mean)
        mse_list.append(eval_mse)
        mae_list.append(eval_mae)
        mape_list.append(eval_mape)

        plotMetric(loss_list, labels, "Loss", str(size))
        plotMetric(valLoss_list, labels, "Val Loss", str(size))
        plotMetric(MSE_list, labels, "MSE", str(size))
        plotMetric(MAPE_list, labels, "MAPE", str(size))

    mean_list, mse_list, mae_list, mape_list
    plotEvals(mean_list, [256, 512, 1024, 2048, 4096, 8192, 16384], "Eval Mean")
    plotEvals(mse_list, [256, 512, 1024, 2048, 4096, 8192, 16384], "Eval MSE")
    plotEvals(mae_list, [256, 512, 1024, 2048, 4096, 8192, 16384], "Eval MAE")
    plotEvals(mape_list, [256, 512, 1024, 2048, 4096, 8192, 16384], "Eval MAPE")

if __name__ == "__main__":
    main()