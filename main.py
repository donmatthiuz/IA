from Connect4 import ConnectFourGameAI,ConnectFourGameHuman
def main():
    while True:
        
        print("""
        ╔════════════════════════╗
        ║     Connect Four       ║
        ╠════════════════════════╣
        ║ 1. 2 jugadores         ║
        ║ 2. Jugar solo          ║
        ║ 3. Ver IAs jugar       ║
        ║ 4. Salir               ║
        ╚════════════════════════╝
        """)
        choice = input("Selecciona una opción: ")
        if choice == "1":
            game = ConnectFourGameHuman()
            game.run()
        if choice == "2":
            game = ConnectFourGameAI()
            game.run()
        elif choice == "4":
            print("Saliendo del juego...")
            break
        else:
            print("Opción inválida. Intenta de nuevo.")
            input("Presiona Enter para continuar...")


if __name__ == "__main__":
    main()