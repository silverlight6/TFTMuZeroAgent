import game_round
import AI_interface

# AI_I = AI_interface.TFT_AI()
# AI_I.train()
AI_interface.train_model()
# error = game_round.human_game_logic()
# if not error: print("An error occurred within the game")

"""
import champion
from champion_functions import MILLIS

import multiprocessing

import tkinter as tk
master = tk.Tk()
tk.Label(master, text="Team data").grid(row=0)
tk.Label(master, text="Iterations").grid(row=1)
tk.Label(master, text="Results").grid(row=2)
tk.Label(master, text='Status').grid(row=3)

team_json = tk.Entry(master)
iterations = tk.Entry(master)
blue_results = tk.Entry(master)
red_results = tk.Entry(master)
bugged_out_results = tk.Entry(master)
draw_results = tk.Entry(master)
status = tk.Entry(master)

team_json.grid(row=0, column=1)
iterations.grid(row=1, column=1)
blue_results.grid(row=2, column=1)
red_results.grid(row=2, column=2)
bugged_out_results.grid(row=2, column=3)
draw_results.grid(row=2, column=4)
status.grid(row=3, column=1)

#team_data = team_json.get()
#iterations_data = iterations.get()

def set_status(val):
    status.delete(0, tk.END)
    status.insert(0, val)
set_status('idle')

def run():

    #global test_multiple
    set_status('running')

    team_data = team_json.get()
    iterations_data = int(iterations.get())

    jobs = []
    
    if(team_data):
        for i in range(1, iterations_data + 2):
            if(status.get() == 'idle'): break
            
            try:
                champion.run(champion.champion, team_data)
            except:
                champion.test_multiple['bugged out'] += 1
           



            blue_results.delete(0, tk.END)
            blue_results.insert(0, 'blue: ' + str(champion.test_multiple['blue']))
            red_results.delete(0, tk.END)
            red_results.insert(0, 'red: ' + str(champion.test_multiple['red']))
            bugged_out_results.delete(0, tk.END)
            bugged_out_results.insert(0, 'bugged rounds: ' + str(champion.test_multiple['bugged out']))
            draw_results.delete(0, tk.END)
            draw_results.insert(0, 'draws: ' + str(champion.test_multiple['draw']))
            master.update()

            with open('log.txt', "w") as out:
                if(MILLIS() < 75000):
                    if(champion.log[-1] == 'BLUE TEAM WON'): champion.test_multiple['blue'] += 1
                    if(champion.log[-1] == 'RED TEAM WON'): champion.test_multiple['red'] += 1
                elif(MILLIS() < 200000): champion.test_multiple['draw'] += 1
                for line in champion.log:
                    out.write(str(line))
                    out.write('\n')
            out.close()


    champion.test_multiple = {'blue': 0, 'red': 0, 'bugged out': 0, 'draw': 0}
    set_status('idle')
    master.update()



    #set_stop(False)
    

start_simulations = tk.Button(master, text='Run simulations', command= lambda: run())   
start_simulations.grid(row=4, column=1, sticky=tk.W, pady=4)

stop_simulations = tk.Button(master, text='Stop', command= lambda:  set_status('idle'))
stop_simulations.grid(row=4, column=2, sticky=tk.W, pady=4)

quit_simulations = tk.Button(master, text='Quit', command= lambda:  master.quit())
quit_simulations.grid(row=4, column=3, sticky=tk.W, pady=4)

tk.mainloop()
"""
