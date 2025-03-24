from Connect4 import ConnectFourGameAI,ConnectFourGameHuman, ConnectFourGameAI_vsAI, Entrenamiento,ConnectFourGameAI, ConnectFourGameAI_train, TDAgent, ConectFourGame_TDAg_IA

import matplotlib.pyplot as plt
import numpy as np

def main():
    while True:

        print("""
        ╔═════════════════════════════╗
        ║     Connect Four            ║
        ╠═════════════════════════════╣
        ║ 1. 2 jugadores              ║
        ║ 2. Jugar solo               ║
        ║ 3. Ver IAs jugar (aburrido) ║
        ║ 4. Tunear TDA               ║
        ║ 5. Resultados               ║
        ║ 6. Salir                    ║
        ╚═════════════════════════════╝
        """)
        choice = input("Selecciona una opción: ")
        if choice == "1":
            game = ConnectFourGameHuman()
            game.run()
        if choice == "2":
            print("""
                ╔═══════════════════════════════╗
                ║     Connect Four              ║
                ╠═══════════════════════════════╣
                ║ 1. Usar poda (modo insano)    ║
                ║ 2. No usar la poda (modo nena)║
                ╚═══════════════════════════════╝
           """)
            choice2 = input("Selecciona una opción: ")
            game = None
            select = False
            if choice2 == "1":
                select = True
            elif choice2 == "2":
                select =  False
            game = ConnectFourGameAI(play_pruning=select)
            game.run()
        if choice == "3":
            print("""
                ╔═══════════════════════════════╗
                ║     Connect Four              ║
                ╠═══════════════════════════════╣
                ║ 1. MINIMAX                    ║
                ║ 2. MINIMAX ALPHA BETA         ║
                ║ 3. TD                         ║  
                ╚═══════════════════════════════╝
           """)
            choice2 = input("Seleccione contra que IA competira nuestro TD: ")

            select = ""
            if choice2 == "1":
                select = "MINIMAX"
            elif choice2 == "2":
                select =  "MINIMAX_ALPHA"
            elif choice2 == "3":
                select =  "TD"
            game = ConectFourGame_TDAg_IA(competir_contra=select)
            game.run()
        elif choice == "6":
            print("Saliendo del juego...")
            break
        elif choice == "4":
            print("Tunear modelo advertencia se tarda mucho")
            tda_agent = TDAgent()
            alpha_range = [0.1, 0.05, 0.01]
            gamma_range = [0.9, 0.8, 0.7]
            epsilon_range = [0.9,  0.5, 0.1]
            tda_agent.fine_tune_parameters(alpha_range=alpha_range, 
                                        gamma_range=gamma_range, 
                                        epsilon_range=epsilon_range)

            break
        elif choice == "5":
            
            print("Mostrando resultados")
            game_td = ConectFourGame_TDAg_IA(competir_contra="TD")
            game_td.analize()
          

            game_mini = ConectFourGame_TDAg_IA(competir_contra="MINIMAX")
            game_mini.analize()

            game_mini2 = ConectFourGame_TDAg_IA(competir_contra="MINIMAX_ALPHA")
            game_mini2.analize()

            def plot_results(victories=None):
                if victories is None:
                    victories = {
                        "TD": game_td.victorias,
                        "MINIMAX": game_mini.victorias,
                        "MINIMAX_ALPHA": game_mini2.victorias,
                    }
                
                labels = list(victories.keys())
                values = list(victories.values())
                
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.bar(labels, values, color=["blue", "green", "red"])
                
                ax.set_ylabel("Victorias")
                ax.set_title("Comparación de Victorias por Competencia de IA")
                
                plt.show()

            plot_results()




        else:
            print("Opción inválida. Intenta de nuevo.")
            input("Presiona Enter para continuar...")


if __name__ == "__main__":
    main()