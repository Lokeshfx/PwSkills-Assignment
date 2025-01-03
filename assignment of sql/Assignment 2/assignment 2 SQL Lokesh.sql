use mavenmovies;
show tables;
show tables;


-- Basic Aggregate Functions: 
-- Question 1: 
-- Retrieve the total number of rentals made in the Sakila database
-- Solution :-
select count(rental_id) from rental;


-- 2. Find the average rental duration (in days) of movies rented
-- Solution:-
select avg(rental_duration) from film;

-- STRING FUNCTIONS:-
-- 3. Display the first name and last name of customers in uppercase. 
-- Solution:-
select upper(first_name), upper(last_name) from customer;


-- 4. Extract the month from the rental date and display it alongside the rental ID. 
-- Solution :-
select rental_id, month(rental_date) from rental;


-- 5. Retrieve the count of rentals for each customer (display customer ID and the count of rentals).
-- Solution :-
select customer_id , count(rental_id) from rental group by customer_id ;

 
select * from store; -- store_id,, manager_staff_id
-- 6. Find the total revenue generated by each store. 
-- Solution:-
SELECT 
    store_id, COUNT(amount) AS Total_revenue_of_store
FROM
    store
        JOIN
    payment ON store.store_id = payment.staff_id
GROUP BY store_id ;


-- 7. Display the title of the movie, customers first name, and last name who rented it. 
/* For Hint-
select *from rental ; -- For:- rental_id, customer_id, inventory_id
select *from customer; -- For:-first_name, last_name, customer_id,
select *from film; -- For :- title of the movie, film_id
select *from inventory; -- For:- film_id , invetory_id
*/
 
SELECT 
    first_name, last_name, title AS Title_of_the_Movie
FROM
    rental
        LEFT JOIN
    customer ON rental.customer_id = customer.customer_id
        LEFT JOIN
    inventory ON rental.inventory_id = inventory.inventory_id
        LEFT JOIN
    film ON inventory.film_id = film.film_id;


-- 8. Retrieve the names of all actors who have appeared in the film "Gone with the Wind".
-- solution:-
/* For Hint-
select *from actor ;-- actor_id
select *from film ; -- film_id, title name
select *from film_actor; -- actor_id, film_id 
*/

SELECT 
    first_name, last_name, title
FROM
    actor
        INNER JOIN
    film_actor ON actor.actor_id = film_actor.actor_id
        INNER JOIN
    film ON film.film_id = film_actor.film_id
WHERE
    title LIKE 'Gone with the Wind';
    

-- GROUP BY 
-- Question 1. Determine the total number of rentals for each category of movies. 
-- Solution:-
/* for Hint-
select *from rental; -- rental_id , customer_id, inventory_id, -
select *from film_category; -- film_id, category_id -
select *from inventory; -- film_id, inventory_id -
select *from category; -- category_id , name of film 
*/
SELECT 
    COUNT(rental_id) AS Total_no_of_rentals, name
FROM
    rental
        JOIN
    inventory ON rental.inventory_id = inventory.inventory_id
        JOIN
    film_category ON film_category.film_id = inventory.film_id
        JOIN
    category ON category.category_id = film_category.category_id
GROUP BY name
ORDER BY Total_no_of_rentals DESC;
 

-- Question 2. Find the average rental rate of movies in each language. 
-- Solution:-
/*
select * from film ;-- rental_rate, language_id, film_id, movieName(title) 
select *from language; -- language_id, name of movie
*/
SELECT 
		avg(rental_rate) as avg_rental_rate_of_movie , name
FROM
    film
        JOIN
    language ON film.language_id = language.language_id group by name ;



-- JOINS
-- Question:- 3. Retrieve the customer names along with the total amount they've spent on rentals.
-- Solution:-
/* hint -
select *from customer; -- customer_id, name
select *from payment; -- customer_id, amount
select *from payment;
select *from rental;
*/

SELECT 
    CONCAT(first_name,' ', last_name) AS customer_name,
    SUM(amount) AS Total_Rental_spent
FROM
    customer
    join rental on rental.customer_id = customer.customer_id
        JOIN
    payment ON customer.customer_id = payment.customer_id
    
GROUP BY customer.customer_id;


-- Question 4. List the titles of movies rented by each customer in a particular city (e.g., 'London'). 
-- Solution:-
/* hint
select *from rental; -- rental_id, customer_id, inventory_id -
select *from film ; -- title of movie, film_id -
select *from inventory; -- film_id, inventory_id -
select *from customer; -- address_id, customer_id -
select *from city ; -- city_id, city
select *from address ; -- city_id, address_id -
*/

select concat(first_name,' ',last_name)as Customer_name , title,city from rental join customer on rental.customer_id = customer.customer_id
join inventory on rental.inventory_id = inventory.inventory_id join film on inventory.film_id = film.film_id
join address on customer.address_id = address.address_id join city on address.city_id = city.city_id ;


-- Advanced Joins and GROUP BY:
-- Question 5. Display the top 5 rented movies along with the number of times they've been rented. 
-- Solution:-
/* for Hint
select * from rental order by customer_id ; -- rental_id , customer_id, inventory_id
select *from film; -- film_id, title
select *from inventory; -- inventory_id, film_id
*/
SELECT 
    COUNT(rental.customer_id) AS Num_of_times,
    title AS Rented_movies
FROM
    rental
        JOIN
    inventory ON rental.inventory_id = inventory.inventory_id
        JOIN
    film ON inventory.film_id = film.film_id
GROUP BY title
ORDER BY Num_of_times DESC
LIMIT 5;


-- Question 6. Determine the customers who have rented movies from both stores (store ID 1 and store ID 2). 
-- Solution:-
select *from rental order by customer_id; -- rental_id, customer_id, inventory_id
select * from customer ; -- customer_id , store_id

SELECT first_name, last_name, email
FROM customer
JOIN rental ON customer.customer_id = rental.customer_id
join inventory on rental.inventory_id = inventory.inventory_id
join store on inventory.store_id = store.store_id
WHERE store.store_id in (1,2)
    GROUP BY customer.customer_id
    HAVING COUNT(DISTINCT store.store_id) = 2;



