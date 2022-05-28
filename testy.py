import random
import numpy as np

# tab = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# tab2 = [*range(10)]
#
# print(tab)
# print(tab2)
#
# # 1
# lista = np.array([[1, 2, 3]] * 10)
# print(lista)


# tworzenie tablicy 10 elementowej randint(start, stop, wielkość tablicy)
# lst = np.random.randint(100, 120, 10)
#

print("------------------------------------")
# 1 sposob
lst = list(np.random.randint(100, 120, 15))
print(lst, "- lista z naszymi wylosowanymi elementami")

# 2 sposob
lst = []
for _ in range(15):
    lst.append(random.randrange(100, 120))
# wypisz zawartość listy
print(lst, "- lista z naszymi wylosowanymi elementami")
print(len(lst), "- długość listy lst")

# wylosuj numer elementu tablicy
nasz_wylosowany_numer = random.randrange(len(lst))
print(nasz_wylosowany_numer, "- numer wysolowoany z zakresu od 0 do długości listy")

# sprawdz czesc od elementu do konca listy
element_do_konca_listy = lst[nasz_wylosowany_numer:]
print(element_do_konca_listy, "- część listy od wybranego elementu")

print("__________DZIELENIE MODULO______________")
print(10 % 2)
print("________________________________________")
# Wypisz parzyste liczby i ich indexy postać-(wartość-index)
for i in range(len(lst)):
    if lst[i] % 2 == 0:
        print(lst[i], " - ", i)

# sprawdz ile jest liczb że suma cyfr wieksza od 10
counter = 0
print("+++++++++++++++++++")

for l in lst:
    templist = list(str(l))
    # print(templist)
    suma_liczb = sum([int(i) for i in templist])
    # print(l, suma_liczb)
    if suma_liczb > 10:
        counter += 1  # counter = counter + 1

print("counter = ", counter)


print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

# sprawdz ile jst takch liczb że suma jest licza pierwszą
counter2 = 0
for l in lst:
    suma_liczb = sum([int(i) for i in list(str(l))])
    # sprawdzanie czy liczba jest pierwsza
    flag = True # flaga gdzie False oznacza, że liczba nie jest pierwsza, a True oznacza, że jest pierwsza
    for i in range(2, suma_liczb):
        # print(suma_liczb, "- suma liczb", i, "- dzielnik")
        if suma_liczb % i == 0:# jeśli liczba jest podzielna przez "i" to wychodzimy z pętli w 71 linijce i ustawiamy flage na False
            flag = False
            break# wyjście całkiem z pętli i nie iterowanie dalej
    if flag == True: # jesli flaga jest true to zwiększamy licznik(counter)
        counter2+=1
        print(suma_liczb, "- ta liczba jest pierwsza")
print("!!!!!!!!!!!!!!!!!!!!!!")
print(counter2, "- tyle jest liczb piwerszych")