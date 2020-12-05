# -*- coding: utf-8 -*-
import random
import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
from sklearn.preprocessing import StandardScaler
from copy import copy

# 可視化系
import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pandas.plotting import scatter_matrix
import plotly.offline as offline
import plotly.graph_objs as go

class GeneticAlgorithm():
    
    def get_domain(self, DF,  exp):
        '''定義域を自動で獲得する関数。
        指定を受けない列はこれによって最小値と最大値が定義域となる
        自分で作る場合も参考にして使ってください。'''
        
        # 一旦カテゴリー変数とダミー変数を考えていないのであとで追加。
        # 全部作るかonlyなので、一部だけ指定する機能を追加予定
        DF = pd.DataFrame(DF)
        domain={}
        for col in exp:
            column = DF.iloc[:, col]
            domain[col] = ('uniform',(min(column), max(column)))
        return domain
    
    def _mutDomain(self, individual, domain_list, indpb=0.05):
        '''GAの突然変異関数。domainの定義域通りに突然変異をする。
        一応どのような突然変異も設定可能。
        domain_listはdomain.valuesをリストにして、choice2のindexだけを抜いたもの。
        '''
        
        for i in range(len(domain_list)):
            if random.random() < indpb:
                # individualがi+1なのはdummyの影響。deapがデバックしたら、i　に戻す。
                if domain_list[i][0]=='randint':
                    individual[i+1] = eval('random.randint{}'.format(domain_list[i][1]))
                elif domain_list[i][0]=='uniform':
                    individual[i+1] = eval('random.uniform{}'.format(domain_list[i][1]))        
                elif domain_list[i][0]=='choice':
                    individual[i+1] = eval('random.choice({})'.format(domain_list[i][1]))        

                elif domain_list[i][0]=='choice2':
                    # choice2の関数が入っている場合
                    #if isinstance(domain_list[i][1], tuple) or isinstance(domain_list[i][1],list):
                        
                    individual[i+1] = eval('random.choice({})'.format(domain_list[i][1]))  
                    # choice2のindexが入っている場合
                    #else:
                    #    pass

                elif domain_list[i][0]=='randrange':
                    individual[i+1] = eval('random.randrange{}'.format(domain_list[i][1])) 

        return individual,

    def _cxTwoPointCopy(self, ind1, ind2):
        '''GAの2点交差の関数。numpy.ndarrayなので作成'''
        size = len(ind1)
        cxpoint1 = random.randint(1, size)
        cxpoint2 = random.randint(1, size - 1)
        if cxpoint2 >= cxpoint1:
            cxpoint2 += 1
        else: # Swap the two cx points
            cxpoint1, cxpoint2 = cxpoint2, cxpoint1
    
        ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] = ind2[cxpoint1:cxpoint2].copy(), ind1[cxpoint1:cxpoint2].copy()
            
        return ind1, ind2
    
    
    def _individual_flatten(self,individual):
        '''individualの評価の時に、choice2でタプルが含まれているのをフラットにする関数'''
        ind = []
        ind_append = ind.append
        for inner in individual:
            if isinstance(inner, tuple) or isinstance(inner, list):
                for flatten in inner:
                    ind_append(flatten)
            else:
                ind_append(inner)
        return np.array([ind])  
    
    def _evaluate_equality(self, individual, model_list, weights=None, obj_scaler=None):
        '''多目的関数の場合に個体を評価する関数(デフォルト)。目的変数をそのまま評価関数として返す。
        model_list : 機械学習のモデルが入ったもの.[[exp→mid],[mid→obj]]の形式。
        midがNoneの場合は[exp→mid]にNoneが入る。
        weights : _evaluate_specificと引数を合わせるためのもの。
        obj_scaler : _evaluate_specificと引数を合わせるためのもの。
        '''
        # individualを相関によって切った場合に説明変数を選択する部分は未実装
        # individualから、一行のDFに展開
        individual = np.delete(individual, 0)  # 一番最初のdummyを落とす
        individual = self._individual_flatten(individual)  # choice2のタプルを平らにする
        if model_list[0]==None:  # midがNoneの場合

            obj_eval = np.empty((1,0))
            for i in range(len(model_list[1])):
                obj_eval = np.append(obj_eval, model_list[1][i].predict(individual) )
        else:
            mid = np.empty((1,0))
            for i in range(len(model_list[0])):
                mid = np.append(mid, model_list[0][i].predict(individual) )
            obj_eval = np.empty((1,0))
            for j in range(len(model_list[1])):
                obj_eval = np.append(obj_eval, model_list[1][j].predict(mid.reshape(1,-1)))
        return tuple(obj_eval.flatten())
    
    def _evaluate_specific(self, individual, model_list, weights, obj_scaler):
        '''重みが複雑な場合に個体を評価する関数(デフォルト)。目的変数をStandardScalerして、重みをかけて合計。
        model_list : 機械学習のモデルが入ったもの.[[exp→mid],[mid→obj]]の形式。
        midがNoneの場合は[exp→mid]にNoneが入る。
        scl : objの目的変数を等しく評価するために、標準化するやつ.fitさせたStandardScalerを想定。
        '''
        # individualを相関によって切った場合に説明変数を選択する部分は未実装
        
        # individualから、一行のDFに展開
        individual = np.delete(individual, 0)  # 一番最初のdummyを落とす
        individual = self._individual_flatten(individual)  # choice2のタプルを平らにする
        
        if model_list[0]==None:  # midがNoneの場合
            
            obj_eval = np.empty((1,0))
            for i in range(len(model_list[1])):
                obj_eval = np.append(obj_eval, model_list[1][i].predict(individual) )

            
        else:  # exp→mid→objとかます場合。
            mid = np.empty((1,0))
            for i in range(len(model_list[0])):
                mid = np.append(mid, model_list[0][i].predict(individual) )
            obj_eval = np.empty((1,0))
            for j in range(len(model_list[1])):
                obj_eval = np.append(obj_eval, model_list[1][j].predict(mid.reshape(1,-1)))
                
        obj_eval = obj_scaler.transform([obj_eval])
        weights_eval = obj_eval*weights
        specific_eval = sum(weights_eval.flatten())    
            
        return  specific_eval,

        
        
    def inverse_fit(self, DF,domain='default',
                    NPOP=100, NGEN=50, POP_SAVE=True, weights='equality', evaluate_function='default', 
                    model_list='self', exp='self', obj='self', mid='self'):
        '''逆計算をする関数
        DF : 計算するデータフレーム
        domain : GAでexpが取りうる値を辞書型で設定。デフォルトではexpの最小値と最大値のuniformになる。
        evaluate : 最適化を計算する関数。デフォルトでは目的変数のStandardScalerの等重み多目的最適化となる。
        weight : 最適化関数の重み。デフォルトでは等重みの最大化(1.0,1.0,...)が課される。
        なお、ダミー変数、カテゴリ変数もどの列か情報が入っていればうまく値をとるように設定
        exp : 実験条件の列を数字のリストで指定(のちに列名も対応)。
        obj : 目的変数の列を数字のリストで指定(のちに列名も対応)。
        mid : 中間生成変数の列を数字のリストで指定(のちに列名も対応)。    
        デフォルトではモデルに保存された情報を使用。
        '''
        # 適当に返すエラーコードは適当です。後にちゃんと意味をつけたりしようかと思います。
        
        # exp, obj, mid等を指定しない場合はモデルをFitした時に使ったものを使用
        if exp=='self':
            exp = self.exp
        if obj=='self':
            obj = self.obj
        if mid=='self':
            mid = self.mid
        
        # DFと、exp,obj,mid列のdfを読み込み
        DF = pd.DataFrame(DF)
        if all([type(x) == int or type(x)==float for x in exp]):
            exp_df = DF.iloc[:, exp]
        elif all([type(x) == str for x in exp]):
            exp_df = DF.loc[:, exp]
        else:
            raise ValueError("Use same type or correct type in obj list")
        
        if all([type(x) == int or type(x)==float for x in obj]):
            obj_df = DF.iloc[:, obj]
        elif all([type(x) == str for x in obj]):
            obj_df = DF.loc[:, obj]
        else:
            raise ValueError("Use same type or correct type in obj list")
        
        if not mid==None:
            if all([type(x) == int for x in mid]):
                mid_df = DF.iloc[:, mid]
            elif all([type(x) == str for x in mid]):
                mid_df = DF.loc[:, mid]
            else:
                print('Use same type or correct type in obj list')
                return 201
        
        
        # デフォルトの定義域domainの作成
        if domain=='default':
            domain = self.get_domain(DF, exp)
        self.domain = domain
        #self.obj_domain = self.get_domain(DF, obj)
        
        # domainの中でchoice2のindexだけの奴を抜いたリストを作ります。
        # _mut_domainの突然変異関数で使います。
        domain_ = list(domain.values())
        domain_list = []
        domain_list_append = domain_list.append
        for i, dom in enumerate(domain_):    
            if domain_[i][0]=='choice2':
                # choice2の関数が入っている場合
                if isinstance(domain_[i][1], tuple) or isinstance(domain_[i][1],list):
                    domain_list_append(dom)
            else:
                domain_list_append(dom)
        
        
        # デフォルトのモデルリスト
        if model_list=='self':
            model_list = self.model_list
        
        # obj列をStandardScalerする。GAの評価関数で使用。
        obj_scaler = StandardScaler()
        obj_scaler.fit(obj_df)
        
        # デフォルトは全て1.0の等重みで作成
        if weights=='equality':
            weights = tuple([1.0 for i in range(len(obj))])
        
        # 一応weightsの原型を保存します。
        self.weights = weights

        # 多目的最適化は(1.0, 100, -0.1)とかしても(1.0, 1.0, -1.0)のように
        # 変換されてしまう。もし、weightsで1.0以外の値が入ってきた場合は、
        # その割合をかけた単目的最適化にして、重みを考えてあげる。
        if evaluate_function=='default':
            if all([abs(x) ==1.0 or x ==0.0 for x in weights]):
                evaluate_function = self._evaluate_equality
            else:
                evaluate_function = self._evaluate_specific
        
        if all([abs(x) ==1.0 or x ==0.0 for x in weights]):
            specific_weight = None
        else:
            specific_weight = copy(weights)
            weights = (1.0, )
        
        # ---------------------------- GAの染色体を作る関数の作成---------------------------------
        
        creator.create('FitnessMulti', base.Fitness, weights=weights )
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMulti)
        
        toolbox = base.Toolbox()
        
        # domainの定義域に沿って、染色体を作成する関数を作成
        #choice2_key_save=-1
        for i in domain.keys():
            if type(domain[i]) in (str, int, float):
                # {1:3}とかよくわからんことになってた場合。
                print("Please Enter correct value at domain in {}".format(i))
                return 202
            
            else:
                if domain[i][0]=='randint':
                    toolbox.register('exp{}'.format(i), 
                            random.randint, int(domain[i][1][0]), int(domain[i][1][1]))
                    
                elif domain[i][0]=='uniform':
                    toolbox.register('exp{}'.format(i), 
                            random.uniform, domain[i][1][0], domain[i][1][1])
                    
                elif domain[i][0]=='choice':
                    toolbox.register('exp{}'.format(i), 
                            random.choice, domain[i][1])   
                    
                elif domain[i][0]=='choice2':
                    # choice2の関数が入っている場合
                    if isinstance(domain[i][1], tuple) or isinstance(domain[i][1],list):
                        toolbox.register('exp{}'.format(i), 
                            random.choice, domain[i][1])
                    # choice2のindexが入っている場合
                    elif isinstance(domain[i][1], int) or isinstance(domain[i][1],float):
                        
                        # choice2の組で、変数同士が離れていても大丈夫にしようかと思ったけど、めんどいからやめた。
                        COMMENTOUT="""
                        def real_value_choice2(key , value_index):
                            '''choice2のkeyがある場所と、組の中の位置を
                            型判定ですぐわかるように複素数で書く'''
                            return complex(key , value_index)
                        
                        # indexの中のどこの位置にあるか。をvalue_indexに格納する。
                        # 4:('choice2', 3)の3の部分が同じかどうかで、位置を格納してく　
                        if domain[i][1]==choice2_key_save:
                            value_index += 1
                        else:
                            value_index = 1
                        choice2_key_save=copy(domain[i][1])
                        toolbox.register('exp{}'.format(i), real_value_choice2, domain[i][1],
                                        value_index)
                        """
                        pass
                    else:
                        print("Please Enter correct value at domain in {}".format(i))
                        return 202
                    
                elif domain[i][0]=='randrange':
                    toolbox.register('exp{}'.format(i), 
                            random.randrange, domain[i][1][0],domain[i][1][1],domain[i][1][2])
                else:
                    print('It is an unexpected value')
                    return 202
                
        # ----------------------------GAの個体と個体群を作る関数の作成-----------------------------
        
        # なぜかrandom.choiceで、タプルを一番最初に入れないと
        # 後にrancom.choiceを入れた時にエラーがでる。
        # deapのGithubのissueに書いておいたのでそのうち治ると思うけど、治るまではこれで。
        # defining attributes for individual
        toolbox.register("dummy", 
                         random.choice, ((0,0),(0,0)) )
        
        individual_function=[toolbox.dummy]
        for i in domain.keys():
            # choice2の本体じゃないときだけ追加
            if not ((domain[i][0]=='choice2') and (isinstance(domain[i][1], int) or isinstance(domain[i][1],float)) ):
                individual_function.append(eval('toolbox.exp{}'.format(i)))
        individual_function = tuple(individual_function) 
        
        toolbox.register('individual', tools.initCycle, creator.Individual,
                         individual_function,
                          n = 1)
        
        # register attributes to individual
        toolbox.register('individual', tools.initCycle, creator.Individual,
                         individual_function,
                          n = 1)
        
        # individual to population
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        
        # -----------------------GAの進化計算の部分の関数を登録----------------------------
        #if evaluate_function=='equality':
        #    evaluate_function=self._evaluate_equality
            
        # evolution
        toolbox.register('mate', self._cxTwoPointCopy)
        toolbox.register('mutate', self._mutDomain, domain_list=domain_list, indpb = 0.05)
        toolbox.register('select', tools.selTournament, tournsize=3)
        toolbox.register('evaluate', evaluate_function,
                         model_list=model_list, weights=specific_weight, obj_scaler=obj_scaler)
        
        # -----------------------GAのメインループ-------------------------------------
        def main(NPOP, NGEN, POP_SAVE=True):
            '''GAのメインループを回す関数。また、この関数内の変数を使いたいので関数を分けない。
            ga.algorithmsでやらないのは、全popを保存したいので。'''
            
            # 決まった乱数を出すように設定
            random.seed(64)
            # 初期の個体群を生成
            pop = toolbox.population(n=NPOP)
            CXPB, MUTPB, NGEN = 0.5, 0.05, NGEN # 交差確率、突然変異確率、進化計算のループ回数
            
            pop_save = []
            if POP_SAVE:
                pop_save_append = pop_save.append
                pop_save_append(copy(pop))

            print("Start of evolution")
        
            # 初期の個体群の評価
            fitnesses = list(map(toolbox.evaluate, pop))
            for ind, fit in zip(pop, fitnesses):
                ind.fitness.values = fit
        
            print("  Evaluated %i individuals" % len(pop))
            
            # 進化計算開始
            for g in range(NGEN):
                print("-- Generation %i --" % g)
        
                # 次世代の個体群を選択
                offspring = toolbox.select(pop, len(pop))
                # 個体群のクローンを生成
                offspring = list(map(toolbox.clone, offspring))
        
                # 選択した個体群に交差と突然変異を適応する
                # 偶数番目と奇数番目の個体を取り出して交差([::2]で偶数、[1::2]で奇数)
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if random.random() < CXPB:
                        toolbox.mate(child1, child2)
                        # 交換した個体はfitness.values(評価値)を削除
                        del child1.fitness.values
                        del child2.fitness.values
        
                for mutant in offspring:
                    if random.random() < MUTPB:
                        toolbox.mutate(mutant)
                        del mutant.fitness.values
        
                # 適合度が計算されていない個体を集めて適合度を計算
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = map(toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit
        
                print("  Evaluated %i individuals" % len(invalid_ind))
    
                # 次世代群をoffspringにする
                pop[:] = offspring
                
                if POP_SAVE:
                    pop_save_append(copy(pop))
                
                # すべての個体の適合度を配列にする
                fits = [ind.fitness.values[0] for ind in pop]
                                
                length = len(pop)
                mean = sum(fits) / length
                sum2 = sum(x*x for x in fits)
                std = abs(sum2 / length - mean**2)**0.5
        
                print("  Min %s" % min(fits))
                print("  Max %s" % max(fits))
                print("  Avg %s" % mean)
                print("  Std %s" % std)
        
            print("-- End of (successful) evolution --")
        
            best_ind = tools.selBest(pop, 1)[0]
            print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
            return best_ind,best_ind.fitness.values, pop, pop_save

        best_ind, best_ind_value, pop, pop_save  = main(NPOP=NPOP,NGEN=NGEN,POP_SAVE=POP_SAVE)
        
        return best_ind, best_ind_value, pop, pop_save
    



    def get_pop_values(self, pop_save, weights='self', model_list='self',index=False):
        '''pop_saveの評価値を配列として取得する。
        weightsはweightsに-1.0or1.0or0.0以外重み付けしたかどうかで、
        重み付けをした場合は単目的の値なので、元のobjの値を復元する。'''
        values_list = []
        values_list_append = values_list.append
        if weights=='self':
            if all([abs(x) ==1.0 or x ==0.0 for x in self.weights]):  # 単目的
                weights=False
            else:  # 多目的
                weights=True

        if weights==False:  # 多目的最適化の場合
            for i, pop in enumerate(pop_save):
                for j, ind in enumerate(pop):
                    if index:
                        values_list_append([i, j] + list(ind.fitness.values) )
                    else:
                        values_list_append(list(ind.fitness.values) )
            values_list = np.array(values_list)
            sum_ = np.sum(values_list[:,2:], axis=1)  # 各値の合計値
            values_list = np.concatenate([values_list, sum_.reshape(-1,1)], axis=1 )
            
        else:  # 重み付けして単目的最適化の場合
            if model_list=='self':
                model_list = self.model_list
            for i, pop in enumerate(pop_save):
                for j, ind in enumerate(pop):
                    if index:
                        values_list_append([i,j] + list(self._evaluate_equality(ind, model_list)) \
                                       + list(ind.fitness.values))
                    else:
                        values_list_append(list(self._evaluate_equality(ind, model_list)) \
                                       + list(ind.fitness.values))
                        
            values_list = np.array(values_list)
            
        return values_list
    
    def get_pop_list(self, pop_save, index=False):
        '''pop_saveをvalueと同じようなリストとして取得する。'''

        if index:  # indexを入れる場合index分の2個分長くする。
            ncol = self._individual_flatten(np.delete(pop_save[0][0],0)).shape[1] + 2
        else:
            ncol = self._individual_flatten(np.delete(pop_save[0][0],0)).shape[1]
        
        pop_list = np.empty((0, ncol), float)
        for i, pop in enumerate(pop_save):
            for j, ind in enumerate(pop):
                ind = np.delete(ind, 0)  # ダミーの(0,0)を落とします。
                if index:
                    index_and_flatten = np.concatenate([np.array([[i,j]]), self._individual_flatten(ind)], axis=1)
                    pop_list = np.concatenate([pop_list, index_and_flatten], axis=0)
                else:
                    pop_list = np.concatenate([pop_list, self._individual_flatten(ind)], axis=0)
                
        return pop_list

    def Ball2dHeatMap(self, DF, valueL='lastrow', color='cool', size=12, marker='o',alpha=0.4,
                    save=False, savepath='./' ):
        '''各列の相関と、評価値の大きさを色で表示する。
        DFには見たいDF, valueLには評価値を入力する。入力しない場合はDFの最後の列'''
        
        DF = np.array(DF)
        if valueL=='lastrow':
            valueL=DF[:, -1]
            DF = np.delete(DF, obj=-1, axis=1)
        
        # colorで
        normalizedValueL = list( (valueL - min(valueL)) / (max(valueL) - min(valueL)) )
        
        if color=='hot':
            colors = plt.cm.hot_r(normalizedValueL)
            #カラーバー表示用
            colmap = plt.cm.ScalarMappable(cmap=plt.cm.hot_r)
        elif color=='cool':
            colors = plt.cm.cool_r(normalizedValueL)
            colmap = plt.cm.ScalarMappable(cmap=plt.cm.cool_r)
        elif color=='hsv':
            colors = plt.cm.hsv_r(normalizedValueL)
            colmap = plt.cm.ScalarMappable(cmap=plt.cm.hsv_r)
        elif color=='jet':
            colors = plt.cm.jet_r(normalizedValueL)
            colmap = plt.cm.ScalarMappable(cmap=plt.cm.jet_r)
        elif color=='gray':
            colors = plt.cm.gray_r(normalizedValueL)
            colmap = plt.cm.ScalarMappable(cmap=plt.cm.gray_r)
        elif color=='spring':
            colors = plt.cm.spring_r(normalizedValueL)
            colmap = plt.cm.ScalarMappable(cmap=plt.cm.spring_r)
        elif color=='summer':
            colors = plt.cm.summer_r(normalizedValueL)
            colmap = plt.cm.ScalarMappable(cmap=plt.cm.summer_r)
        elif color=='autumn':
            colors = plt.cm.autumn_r(normalizedValueL)
            colmap = plt.cm.ScalarMappable(cmap=plt.cm.autumn_r)
        elif color=='winter':
            colors = plt.cm.winter_r(normalizedValueL)
            colmap = plt.cm.ScalarMappable(cmap=plt.cm.winter_r)
        else:
            print('Since there is no color, it will be the default cool')
            colors = plt.cm.cool_r(normalizedValueL)
            colmap = plt.cm.ScalarMappable(cmap=plt.cm.cool_r)
        
        
        
        colmap.set_array(valueL)
        
        plt.figure()
        
        ax_matrix = scatter_matrix(pd.DataFrame(DF), c=colors, s=size, marker=marker, alpha=alpha)
        
        #カラーバー
        plt.colorbar(colmap, ax=ax_matrix)
        if save==True:
            date = datetime.datetime.now()
            plt.savefig(savepath+'scatter_matrix_'+str(date.year)+'_'+ str(date.month)+ \
                        '_'+str(date.day)+'_'+str(date.hour)+'_'+ \
                        str(date.minute)+'_'+str(date.second), dpi=150)
        plt.show()
    
    

    def Ball3dHeatMap(self, xL, yL ,zL, valueL, grid=True, color='cool',
                    size=100, marker='o',alpha=0.8,save=False, savepath='./'):
        '''3次元での値の評価をみるもの。値は色で表現する。'''
        
        #Normalize valueL into 0 to 1
        normalizedValueL = list( (valueL - min(valueL)) / (max(valueL) - min(valueL)) )
        
        if color=='hot':
            colors = plt.cm.hot_r(normalizedValueL)
            #カラーバー表示用
            colmap = plt.cm.ScalarMappable(cmap=plt.cm.hot_r)
        elif color=='cool':
            colors = plt.cm.cool_r(normalizedValueL)
            colmap = plt.cm.ScalarMappable(cmap=plt.cm.cool_r)
        elif color=='hsv':
            colors = plt.cm.hsv_r(normalizedValueL)
            colmap = plt.cm.ScalarMappable(cmap=plt.cm.hsv_r)
        elif color=='jet':
            colors = plt.cm.jet_r(normalizedValueL)
            colmap = plt.cm.ScalarMappable(cmap=plt.cm.jet_r)
        elif color=='gray':
            colors = plt.cm.gray_r(normalizedValueL)
            colmap = plt.cm.ScalarMappable(cmap=plt.cm.gray_r)
        elif color=='spring':
            colors = plt.cm.spring_r(normalizedValueL)
            colmap = plt.cm.ScalarMappable(cmap=plt.cm.spring_r)
        elif color=='summer':
            colors = plt.cm.summer_r(normalizedValueL)
            colmap = plt.cm.ScalarMappable(cmap=plt.cm.summer_r)
        elif color=='autumn':
            colors = plt.cm.autumn_r(normalizedValueL)
            colmap = plt.cm.ScalarMappable(cmap=plt.cm.autumn_r)
        elif color=='winter':
            colors = plt.cm.winter_r(normalizedValueL)
            colmap = plt.cm.ScalarMappable(cmap=plt.cm.winter_r)
        else:
            print('Since there is no color, it will be the default cool')
            colors = plt.cm.cool_r(normalizedValueL)
            colmap = plt.cm.ScalarMappable(cmap=plt.cm.cool_r)
        
        colmap.set_array(valueL)
     
        fig = plt.figure()
        ax = Axes3D(fig)
     
        #Set the grid on of off
        if not grid:
            ax.grid(False)
     
        ax.scatter(xL,yL,zL, s =size, c=colors, marker=marker, alpha=alpha)
        #カラーバー
        cb = fig.colorbar(colmap)
     
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        if save==True:
            date = datetime.datetime.now()
            plt.savefig(savepath+'3Dheatmap_'+str(date.year)+'_'+ str(date.month)+ \
                       '_'+str(date.day)+'_'+str(date.hour)+'_'+ \
                       str(date.minute)+'_'+str(date.second), dpi=150)
        plt.show()
    

    def plotly3dHeatMap(self, xL, yL , zL, valueL, size=6, alpha=0.8, color='Viridis'):
        '''3次元のヒートマップplotly版'''
        
        # 保存方法については何も設定していないので要検討
        trace1 = go.Scatter3d(
        x=xL,
        y=yL,
        z=zL,
        mode='markers',
        marker=dict(
            size=size,
            color=valueL,                # set color to an array/list of desired values
            colorscale=color,   # choose a colorscale
            opacity=alpha,
            colorbar=dict(thickness=20)
            )
        )
    
        data = [trace1]
        layout = go.Layout(
            margin=dict(
                l=0,
                r=0,
                b=0,
                t=0
            )
        )
        fig = go.Figure(data=data, layout=layout)
        offline.plot(fig, filename='3d-scatter-colorscale')