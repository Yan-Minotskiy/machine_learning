# Обучение языку запросов SQL

![](../image/SQL.png)
## Другие мои работы связанные с этой тематикой
[Практические и лабораторные работы по дисциплине "Клиент-серверные системы управления банком данных"](https://github.com/Yan-Minotskiy/Labs/tree/master/%D0%91%D0%B0%D0%B7%D1%8B%20%D0%B4%D0%B0%D0%BD%D0%BD%D1%8B%D1%85)

## Типы данных
Подробнее о типах данных в SQL можно посмотреть [здесь](https://unetway.com/tutorial/sql-tipy-dannyh).
## Основные операторы
### CREATE

Создание базы данных
```sql
CREATE DATABASE db1;
```

Создание таблицы
```sql
CREATE TABLE users(
    user_id INT NOT NULL AUTO_INCREMENT,
    name VARCHAR(200),
    num_friends INT,
    PRIMARY KEY(user_id));
```

Добавление столбцов в таблицу
```sql
ALTER TABLE table_name
ADD column_name datatype;
```

### INSERT

```sql
INSERT INTO users (user_id, name, num_friends) VALUES (0, 'Hero', 0)
```

Оператор `INSERT INTO SELECT` копирует данные из одной таблицы и вставляет ее в другую таблицу. `INSERT INTO SELECT` требует, чтобы типы данных в исходной и целевой таблицах соответствовали.
```sql
INSERT INTO table2 (column1, column2, column3, ...)
SELECT column1, column2, column3, ...
FROM table1
WHERE condition;
```

### UPDATE

```sql
UPDATE users
SET phone = '+ 7 (777) 000 000 00', email = 'mail@yan.com'
WHERE user_id = 1;
```
Изменить тип данных столбца в таблице можно так
```sql
ALTER TABLE table_name
MODIFY COLUMN column_name datatype;
```
или так
```sql
ALTER TABLE table_name
ALTER COLUMN column_name datatype;
```
### DELETE

Удаление базы данных
```sql
DROP DATABASE databasename;
```

Удаление таблицы
```sql
DROP TABLE table_name;
```
Оператор `TRUNCATE TABLE` используется для удаления данных внутри таблицы, но не самой таблицы.
```sql
TRUNCATE TABLE table_name;
```

Удаление столбцов из таблицы
```sql
ALTER TABLE table_name
DROP COLUMN column_name;
```

### SELECT

Выбрать все столбцы таблицы
```sql
SELECT * FROM users;
```

Выбрать определённые колонки из таблицы
```sql
/* Это многострочный
комментарий */
SELECT name, fullname FROM users;
```

Выбрать определённые колонки из таблицы без дублирования
```sql
-- Это однострочный комментарий
SELECT DISTINCT column1, column2
FROM table_name;
```
Выборка в соответствии с условием прописанного в операторе `WHERE`
```sql
SELECT * FROM users
WHERE name != 'Том';
```

Несколько вариантов выбора первых 3 записей таблицы
```sql
SELECT TOP 3 * FROM users;
```
```sql
SELECT * FROM users
LIMIT 3;
```
```sql
SELECT * FROM users
WHERE ROWNUM <= 3;
```

Выборка первой четверти записей в таблице
```sql
SELECT TOP 25 PERCENT * FROM users;
```
Пример объединения запросов
```sql
SELECT column_name(s) FROM table1
UNION
SELECT column_name(s) FROM table2;
```

Чтобы разрешить повторяющиеся значения, используйте `UNION ALL`.
```sql
SELECT column_name(s) FROM table1
UNION ALL
SELECT column_name(s) FROM table2;
```

Пример использования опратора `HAVING` для работы с агрегаторными функциями
```sql
SELECT COUNT(user_id), country
FROM users
GROUP BY country
HAVING COUNT(user_id) > 7;
```

Оператор `EXISTS` используется для проверки существования любой записи в подзапросе. Если подзапрос возвращает одну или несколько записей, то возвращается true.
```sql
SELECT column_name(s)
FROM table_name
WHERE EXISTS
(SELECT column_name FROM table_name WHERE condition);
```

Оператор `SELECT INTO` копирует данные из одной таблицы в новую таблицу.
```sql
SELECT *
INTO newtable [IN externaldb]
FROM oldtable
WHERE condition;
```


### ORDER BY

Сортировка выборки. `ASC` - по возрастанию, `DESC` - по убыванию.
```sql
SELECT column1, column2, ...
FROM table_name
ORDER BY column1, column2, ... ASC|DESC;
```
### JOIN
![](../image/join.jpg)

Все счета с информацией о пользователях и отправителях
```sql
SELECT invoice.invoice, users.name, addresser.name
FROM ((invoice
INNER JOIN users ON invoice.user_id = users.user_id)
INNER JOIN addresser ON invoice.addresser_id = addresser.addresser_id);
```

Все пользователи и любые заказы, которые они могут иметь.
```sql
SELECT users.name, invoice.invoice_id
FROM users
LEFT JOIN invoice ON users.user_id = invoice.user_id
ORDER BY users.name;
```

Тоже самое, только другим способом
```sql
SELECT invoice.invoice_id, users.name, users.fullname
FROM invoice
RIGHT JOIN users ON invoice.user_id = users.user_id
ORDER BY invoice.invoice_id;
```

Все пользователи и все заказы.
```sql
SELECT users.name, invoice.invoice_id
FROM users
FULL OUTER JOIN invoice ON users.user_id = invoice.user_id
ORDER BY users.name;
```

Следующий оператор SQL соответствует пользователям из одного города (self JOIN).
```sql
SELECT A.name AS name1, B.name AS name2, A.city
FROM users A, users B
WHERE A.user_id <> B.user_id
AND A.city = B.city 
ORDER BY A.city;
```

### GROUP BY

```sql
SELECT column_name(s)
FROM table_name
WHERE condition
GROUP BY column_name(s)
ORDER BY column_name(s);
```

Количество пользователей в каждой стране
```sql
SELECT COUNT(use_id), country
FROM users
GROUP BY country;
```

Количество заказов отправленных каждой службой доставки
```sql
SELECT delivery.name, COUNT(invoice.delivery_id) AS orders FROM invoice
LEFT JOIN delivery ON invoice.delivery_id = delivery.delivery_id
GROUP BY name;
```

### KEYS
`PRIMARY KEY` однозначно идентифицирует каждую запись в таблице базы данных. Первичные ключи должны содержать `UNIQUE` значения и не могут содержать значения `NULL`.
В таблице может быть только один первичный ключ, который может состоять из одного или нескольких полей.
```sql
CREATE TABLE users (
    user_id int NOT NULL,
    name varchar(255) NOT NULL,
    fullname varchar(255),
    gender int,
    PRIMARY KEY (user_id)
);
```

`FOREIGN KEY` - это ключ, используемый для соединения двух таблиц вместе. Является полем (или набором полей) в одной таблице, которое ссылается на `PRIMARY KEY` в другой таблице.
Таблица, содержащая внешний ключ, называется дочерней таблицей, а таблица, содержащая ключ-кандидат, называется ссылочной или родительской таблицей.

```sql
CREATE TABLE invoice (
    invoice_id int NOT NULL,
    number int NOT NULL,
    user_id int,
    PRIMARY KEY (invoice_id),
    CONSTRAINT FK_UserInvoice FOREIGN KEY (user_id)
    REFERENCES Users(user_id)
);
```

Чтобы создать ограничение `FOREIGN KEY` в столбце «user_id», когда таблица «invoice» уже создана, используйте следующее:
```sql
ALTER TABLE invoice
ADD FOREIGN KEY (user_id) REFERENCES Users(user_id);
```
## Логические операторы

| Оператор      | Описание                                                          |
| ------------- |-------------------------------------------------------------------|
| ALL           | Если все значения подзапроса являются TRUE                        |
| AND           | Если все условия, разделенные И, являются TRUE                    |
| ANY           | Если какое-либо из значений подзапроса соответствует TRUE условию |
| BETWEEN       | Если операнд находится в диапазоне сравнения                      |
| EXISTS        | Если подзапрос возвращает одну или несколько записей              |
| IN            | Если операнд равен одному из списка выражений                     |
| LIKE          | Если операнд соответствует шаблону                                |
| NOT           | Отображает запись, если условие (И) НЕ TRUE                       |
| OR            | Если любое из условий, разделенных OR, является TRUE.             |
| SOME          | Если какое-либо из значений подзапроса соответствует условию      |


## SQL - функции
SQL имеет множество встроенных функций: строковые, числовые, даты и расширенные функции. Подробнее почитать о них можно [здесь](https://unetway.com/tutorial/sql-funkcii).

## Нормальные формы баз данных
[Статья на эту тему.](https://office-menu.ru/uroki-sql/51-normalizatsiya-bazy-dannykh)

## Используемые ресурсы
* https://tproger.ru/translations/sql-recap
* Грас Джоэл. "Data Science. Наука о данных с нуля".
* https://stepik.org/course/551
* https://unetway.com/tutorial/sql
* https://office-menu.ru/uroki-sql/51-normalizatsiya-bazy-dannykh
