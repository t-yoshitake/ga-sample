import random
from deap import base, creator, tools
from deap.benchmarks import ackley

# 適応度クラスの作成
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

# 個体クラスの作成
creator.create("Individual", list, fitness=creator.FitnessMin)

# Toolboxの作成
toolbox = base.Toolbox()

# 遺伝子を生成する関数"attr_gene"を登録
toolbox.register("attr_gene", random.uniform, -500, 500)

# 個体を生成する関数”individual"を登録
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_gene, 2)

# 個体集団を生成する関数"population"を登録
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 評価関数"evaluate"を登録
toolbox.register("evaluate", ackley)

# 交叉を行う関数"mate"を登録
toolbox.register("mate", tools.cxBlend, alpha=0.2)

# 変異を行う関数"mutate"を登録
toolbox.register("mutate", tools.mutGaussian, mu=[
                 0.0, 0.0], sigma=[200.0, 200.0], indpb=0.2)

# 個体選択法"select"を登録
toolbox.register("select", tools.selTournament, tournsize=3)


def main():

    random.seed(1)

    # GAパラメータ
    N_GEN = 100
    POP_SIZE = 1000
    CX_PB = 0.5
    MUT_PB = 0.2

    # 個体集団の生成
    pop = toolbox.population(n=POP_SIZE)
    print("Start of evolution")

    # 個体集団の適応度の評価
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    print("  Evaluated %i individuals" % len(pop))

    # 適応度の抽出
    fits = [ind.fitness.values[0] for ind in pop]

    # 進化ループ開始
    g = 0
    while g < N_GEN:

        g = g + 1
        print("-- Generation %i --" % g)

        # 次世代個体の選択・複製
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # 交叉
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # 交叉させる個体を選択
            if random.random() < CX_PB:
                toolbox.mate(child1, child2)

                # 交叉させた個体は適応度を削除する
                del child1.fitness.values
                del child2.fitness.values

        # 変異
        for mutant in offspring:

            # 変異させる個体を選択
            if random.random() < MUT_PB:
                toolbox.mutate(mutant)

                # 変異させた個体は適応度を削除する
                del mutant.fitness.values

        # 適応度を削除した個体について適応度の再評価を行う
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        print("  Evaluated %i individuals" % len(invalid_ind))

        # 個体集団を新世代個体集団で更新
        pop[:] = offspring

        # 新世代の全個体の適応度の抽出
        fits = [ind.fitness.values[0] for ind in pop]

        # 適応度の統計情報の出力
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5

        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)


    print("-- End of (successful) evolution --")

    # 最良個体の抽出
    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))


if __name__ == '__main__':

    main()
