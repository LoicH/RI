

# Information Retrieval course
## at Université Pierre et Marie Curie (Paris VI)

This repo hosts the various search engines developped for the "Information Retrieval" course.

It has 3 parts:

### 1. An accurate text search engine
Find the most relevant scientific article (from 3000 in our database) for a query.

**Example**:
Query: "Number-theoretic algorithms, especially involving prime number series,
sieves, and Chinese Remainder theorem"
Article results:
- A Prime Number Generator Using The Treesort Principle
- An Improved Algorithm to Produce Complex Primes
- Greatest Common Divisor of n Integers and Multipliers

### 2. An image search engine
Find the most relevant images with semantic analysis (for a query such as "tree frog", it's better to return images of Wood Frogs than images of guitars).

<img src="http://4.bp.blogspot.com/-59vl8F0D8lM/TlOjGTaCM9I/AAAAAAAAAcw/QdQPmBIGaNE/s1600/tree_frog_1.jpg" align="left" height="200"> 
<img src="https://nature.mdc.mo.gov/sites/default/files/styles/centered_full/public/media/images/2010/04/wood_frog1.jpg" align="left" height="200">
<br>
<br>
<br>
<br>
<br>
<br>

### 3.  A diverse text search engine
Using diversity to cover all the topics for a query. This search engine avoids redundancy and doesn't return the same image. It returns various diverse images that suit the need of the user. 
