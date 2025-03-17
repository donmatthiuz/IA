from Connect4 import ConnectFourGameAI,ConnectFourGameHuman, ConnectFourGameAI_vsAI
def main():
    while True:
        
        print("""
        ╔═════════════════════════════╗
        ║     Connect Four            ║
        ╠═════════════════════════════╣
        ║ 1. 2 jugadores              ║
        ║ 2. Jugar solo               ║
        ║ 3. Ver IAs jugar (aburrido) ║
        ║ 4. Salir                    ║
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
            game = ConnectFourGameAI_vsAI()
            game.run()
        elif choice == "4":
            print("Saliendo del juego...")
            break
        else:
            print("Opción inválida. Intenta de nuevo.")
            input("Presiona Enter para continuar...")


if __name__ == "__main__":
    main()