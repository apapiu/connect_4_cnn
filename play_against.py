def play_against_human(player_class, human_player):
    winner = None
    s = np.zeros([6,7])
    move_num = 0
    player = 1
    preds = get_nn_preds(s, model, player)
    while winner is None:
        display(plot_state(s))
        preds.plot(kind="bar", figsize = (2.1,2))
        plt.show()
        
        if player == human_player:
            idx = int(input('Choose move number: '))
            s = make_move(s, idx, player)
        else: 
            preds = get_nn_preds(s, model_13, player)
            s = player_class.make_move(s, player)
            time.sleep(0.13)#np.random.normal(1,0.15))
           
        if winning_move(s, player):
            winner = player
            display(plot_state(s))
            preds.plot(kind="bar", figsize = (2.1,2))
            print("Winner is {0}".format(player))
            break
        player *= -1
        move_num += 1
        if move_num == 42:
            winner = 0
        clear_output()


def plot_state(s_new):
    sdf = pd.DataFrame(s_new)
    sdf = sdf.apply(lambda x: x.map({-1.0:"O", 1:"X"}))
    sdf = sdf.sort_index(ascending=False).fillna(".") 
    return sdf.style.applymap(color_negative_red) 

def color_negative_red(val):
    """
    Takes a scalar and returns a string with
    the css property `'color: red'` for negative
    strings, black otherwise.
    """
    if val == "X":
        color = "red"
    elif val == "O":
        color = "blue"
    else:
        color = "grey"
    return 'color: %s' % color


#play_against_human(NNPlayer(optimal_nn_move_noise, model_13, 0, 0), 1)
