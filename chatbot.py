# PA6, CS124, Stanford, Winter 2019
# v.1.0.3
# Original Python code by Ignacio Cases (@cases)
######################################################################
import util

from porter_stemmer import PorterStemmer
import numpy as np
import re


# noinspection PyMethodMayBeStatic
class Chatbot:
    """Simple class to implement the chatbot for PA 6."""

    def __init__(self, creative=False):
        # The chatbot's default name is `moviebot`.
        self.name = 'BOTias'

        self.creative = creative

        # This matrix has the following shape: num_movies x num_users
        # The values stored in each row i and column j is the rating for
        # movie i by user j
        self.titles, ratings = util.load_ratings('data/ratings.txt')
        self.sentiment = util.load_sentiment_dictionary('data/sentiment.txt')
        
        
        # user information
        self.user_counter = 0
        self.user_ratings = np.zeros(( ratings.shape[0],1))

        ########################################################################
        # Binarize the movie ratings matrix.                             #
        ########################################################################

        # Binarize the movie ratings before storing the binarized matrix.
        ratings = self.binarize(ratings)
        self.ratings = ratings
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    ############################################################################
    # 1. WARM UP REPL                                                          #
    ############################################################################

    def greeting(self):
        """Return a message that the chatbot uses to greet the user."""
        ########################################################################
        ########################################################################

        greeting_message = "Wonderful day, innit? How may I be of service?"

        print(self.extract_titles('I thought 10 things i hate about you was great'))

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        return greeting_message

    def goodbye(self):
        """
        Return a message that the chatbot uses to bid farewell to the user.
        """
        ########################################################################
        ########################################################################

        goodbye_message = "The day's end is upon us... farewell!"

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return goodbye_message

    ############################################################################
    # 2. Modules 2 and 3: extraction and transformation                        #
    ############################################################################

    def process(self, line):
        """Process a line of input from the REPL and generate a response.

        This is the method that is called by the REPL loop directly with user
        input.

        You should delegate most of the work of processing the user's input to
        the helper functions you write later in this class.

        Takes the input string from the REPL and call delegated functions that
          1) extract the relevant information, and
          2) transform the information into a response to the user.

        Example:
          resp = chatbot.process('I loved "The Notebook" so much!!')
          print(resp) // prints 'So you loved "The Notebook", huh?'

        :param line: a user-supplied line of text
        :returns: a string containing the chatbot's response to the user input
        """
        ########################################################################
        # TODO: Implement the extraction and transformation in this method,    #
        # possibly calling other functions. Although your code is not graded   #
        # directly based on how modular it is, we highly recommended writing   #
        # code in a modular fashion to make it easier to improve and debug.    #
        ########################################################################
        if (self.user_counter) %5 == 0 and self.user_counter != 0:
            
            
            
            ##################### PLUG IN RECOMMENDATION STUFF HERE
            
            recommendation = "some shitty recommendation"
            
            response = "Given what you told me, I think you would like " + recommendation + ". Would you like more recommendations?"
            return response
            # make a recommendation.
        
        if self.creative:
            response = "I processed {} in creative mode!!".format(line)
        else:

            title_list = self.extract_titles(line)
            
            
#             print("detected movie titles: ",title_list) 
            
            if len(title_list) == 0:
                response = "You didn't even mention a movie to me, you plebian."
                return response
            
            elif len(title_list) == 1:
                
                title = title_list[0]
                
                title_movies = self.find_movies_by_title(title)
                print("same titled movies in database: ", title_movies)
                
                if len(title_movies) == 0:
                    response = "Haven't even heard of what you mentioned in our database. You must be pretty cool to know about it, huh?"
                    return response
                
                if len(title_movies) == 1:
                    
                    movie_id = title_movies[0]
                
                    sentiment_val = self.extract_sentiment(line)
                    self.user_ratings[movie_id] = sentiment_val


                    if sentiment_val == -1:
                        response = "You hate " + title + ", huh?"
                        
                    elif sentiment_val == 0:
                        response = "You're pretty in the middle about " + title + ", aren't you?"
                    else:
                        response = "You just adore " + title + ", don't you?"
                        
                        
                    self.user_counter += 1
                    
                else:
                    response = "I found more than one movie called " + title + ". Can you clarify?"
                    return response
                        
                   
                
            else:
                # more than one movie
                tuples = self.extract_sentiment_for_movies( line )
                                
                mult_response = ""
                
                for i, tup in enumerate(tuples):
                    # first title, then movie_id, then sentiment
                    title, sentiment_val = tup
                    movie_id = self.find_movies_by_title(title)[0]
                    
                    self.user_ratings[movie_id] = sentiment_val
                    
                    if i == len(tuples)-1:
                        # make sure the and is on the last term.
                        mult_response += "and "
                    
                    if sentiment_val == -1:
                        mult_response += "you certainly don't like " + title + ", "
                        
                    elif sentiment_val == 0:
                        mult_response += "you're pretty in the middle about " + title + ", "
                    else:
                        mult_response += "you just looove " + title + ", "
                    
                    if i == len(tuples)-1:
                        # remove the last space and comma.
                        mult_response = mult_response[:len(mult_response)-2]
                    
                
                response = "It seems to me that " + mult_response + ". What a mouthful."

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return response

    @staticmethod
    def preprocess(text):
        """Do any general-purpose pre-processing before extracting information
        from a line of text.

        Given an input line of text, this method should do any general
        pre-processing and return the pre-processed string. The outputs of this
        method will be used as inputs (instead of the original raw text) for the
        extract_titles, extract_sentiment, and extract_sentiment_for_movies
        methods.

        Note that this method is intentially made static, as you shouldn't need
        to use any attributes of Chatbot in this method.

        :param text: a user-supplied line of text
        :returns: the same text, pre-processed
        """
        ########################################################################
        # TODO: Preprocess the text into a desired format.                     #
        # NOTE: This method is completely OPTIONAL. If it is not helpful to    #
        # your implementation to do any generic preprocessing, feel free to    #
        # leave this method unmodified.                                        #
        ########################################################################

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

        return text

    def extract_titles(self, preprocessed_input):
        """Extract potential movie titles from a line of pre-processed text.

        Given an input text which has been pre-processed with preprocess(),
        this method should return a list of movie titles that are potentially
        in the text.

        - If there are no movie titles in the text, return an empty list.
        - If there is exactly one movie title in the text, return a list
        containing just that one movie title.
        - If there are multiple movie titles in the text, return a list
        of all movie titles you've extracted from the text.

        Example:
          potential_titles = chatbot.extract_titles(chatbot.preprocess(
                                            'I liked "The Notebook" a lot.'))
          print(potential_titles) // prints ["The Notebook"]

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: list of movie titles that are potentially in the text
        """
        movie_list = re.findall(r'\"(.+?)\"', preprocessed_input)
        if not self.creative:
            return movie_list

        else:
            preprocessed_input = re.sub(r'[^A-Za-z0-9\. ]+', '', preprocessed_input).lower()

            for movie_data in self.titles:
                title_end_index = movie_data[0].find(' (')
                title = movie_data[0][:title_end_index].lower()
                title_words = title.split()
                new_title = ''
                comma_flag = False
                for i in range(len(title_words)):
                    if "," in title_words[i]:
                        new_title = ' '.join(title_words[i+1:])
                        new_title += ' ' + ' '.join(title_words[0: i+1])
                        comma_flag = True
                        break

                if not comma_flag:
                    new_title = title
                new_title = re.sub(r'[^A-Za-z0-9\. ]+', '', new_title)
                
                if new_title in preprocessed_input:
                    flag = True
                    find_index = preprocessed_input.find(new_title)
                    if find_index != 0 and preprocessed_input[find_index - 1] != ' ':
                        flag = False
                    if find_index + len(new_title) < len(preprocessed_input) and preprocessed_input[find_index + len(new_title)] != ' ':
                        flag = False
                    if flag:
                        movie_list.append(new_title)
                
            return movie_list

    def find_movies_by_title(self, title):
        """ Given a movie title, return a list of indices of matching movies.

        - If no movies are found that match the given title, return an empty
        list.
        - If multiple movies are found that match the given title, return a list
        containing all of the indices of these matching movies.
        - If exactly one movie is found that matches the given title, return a
        list
        that contains the index of that matching movie.

        Example:
          ids = chatbot.find_movies_by_title('Titanic')
          print(ids) // prints [1359, 2716]

        :param title: a string containing a movie title
        :returns: a list of indices of matching movies
        """
        year = re.findall(r'\((\d{4})\)', title)
        # year is a list.
        if year == []:
            year = None
        else:
            year = year[0]

        title = title.lower()
        articles = {"the", "an", "a"}
        title_words = title.split()

        if year is not None:
            title = ' '.join(title_words[:-1])

        if title_words[0] in articles:
            if year is None:
                title = ' '.join(title_words[1:]) + ', ' + title_words[0]
            else:
                title = ' '.join(title_words[1:-1]) + ', ' + title_words[0]

        ids = []
        for i in range(len(self.titles)):
            movie_data = self.titles[i]
            title_end_index = movie_data[0].find(' (')
            movie_title = movie_data[0][:title_end_index]
            if title == movie_title.lower():
                if year is not None:
                    movie_year = re.findall(r'\((\d{4})\)', movie_data[0])
                    if year == movie_year[0]:
                        ids.append(i)
                else:
                    ids.append(i)
        
        return ids


    def extract_sentiment(self, preprocessed_input):
        """Extract a sentiment rating from a line of pre-processed text.

        You should return -1 if the sentiment of the text is negative, 0 if the
        sentiment of the text is neutral (no sentiment detected), or +1 if the
        sentiment of the text is positive.

        As an optional creative extension, return -2 if the sentiment of the
        text is super negative and +2 if the sentiment of the text is super
        positive.

        Example:
          sentiment = chatbot.extract_sentiment(chatbot.preprocess(
                                                    'I liked "The Titanic"'))
          print(sentiment) // prints 1

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: a numerical value for the sentiment of the text
        """
        negation_words = {'not', 'never', "doesn't", "isn't", "wasn't", "shouldn't", "wouldn't", "couldn't", "won't", "can't", "don't", "didn't"}
        stemmer = PorterStemmer()

        first_quote = preprocessed_input.find('"')
        second_quote = preprocessed_input.find('"', first_quote + 1)
        input_without_title = preprocessed_input[:first_quote] + preprocessed_input[second_quote+1:]
        input_without_title = input_without_title.lower()

        pos_count = 0
        neg_count = 0
        flipped_sentiment = False
        for word in input_without_title.split():
            if word in negation_words:
                flipped_sentiment = True

            stemmed_word = stemmer.stem(word, 0, len(word) - 1)
            candidates = [word, stemmed_word]
            # Check past tense edge case.
            if word[-2:] == "ed":
                candidates.append(word[:-1])
                candidates.append(word[:-2])

            for candidate in candidates:
                if self.sentiment.get(candidate, '') == 'pos':
                    if flipped_sentiment:
                        neg_count += 1
                    else:
                        pos_count += 1
                    break
                if self.sentiment.get(candidate, '') == 'neg':
                    if flipped_sentiment:
                        pos_count += 1
                    else:
                        neg_count += 1
                    break

        if pos_count > neg_count:
            return 1
        elif neg_count > pos_count:
            return -1
        else:
            return 0

    def extract_sentiment_for_movies(self, preprocessed_input):
        """Creative Feature: Extracts the sentiments from a line of
        pre-processed text that may contain multiple movies. Note that the
        sentiments toward the movies may be different.

        You should use the same sentiment values as extract_sentiment, described

        above.
        Hint: feel free to call previously defined functions to implement this.

        Example:
          sentiments = chatbot.extract_sentiment_for_text(
                           chatbot.preprocess(
                           'I liked both "Titanic (1997)" and "Ex Machina".'))
          print(sentiments) // prints [("Titanic (1997)", 1), ("Ex Machina", 1)]

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: a list of tuples, where the first item in the tuple is a movie
        title, and the second is the sentiment in the text toward that movie
        """
        # idea is sentiment umbrellas
        # until a word like "but, except, however" and other stoppers like that apply, everything up to that is under one umbrella. 
        # we then just divide the string using those umbrellas, where multiple movies can be under one umbrella
        
        
        # one of the cases where this fucks up is where someone doesn't use grammatically correct stoppers
        # such as, "I hate "superman" and I hate "Ex Machina" and I love "I, Robot".
        # I want to put "and" into the transition phrases, but that fucks up usual things i.e. the first part of that example
        # but it seemed that most of their examples use some of these transition phrases.
        
        flipping_transition_phrases = {'but', 'although', "in contrast", "instead", "whereas", "despite", "otherwise", "however", "regardless", "while", "yet", "on the other hand", "except", "nevertheless", "in contrast"}
        
        neg_transition_inds = [0]            

        for phrase in flipping_transition_phrases:
            if phrase in preprocessed_input:
                neg_transition_inds.append(preprocessed_input.find(phrase))
                
        neg_transition_inds.sort()
        # gonna want ascending indices of negative transitions.
        
        
        
        # now to split up the string
        substrs = []
        
        for i, trans_ind in enumerate(neg_transition_inds):
            
            if i < len(neg_transition_inds)-1:
                substr = preprocessed_input[trans_ind:neg_transition_inds[i+1]]
            else:
                substr = preprocessed_input[trans_ind:]
                
            substrs.append(substr)
                                                                            
        
        
        # our return
        tuples = []
        
        
        for substr in substrs:
            
 
            title_list = self.extract_titles(substr)
            if len(title_list) == 0:
                # no movies mentioned in this substr.
                continue

            sentiment_val = self.extract_sentiment(substr)

            for title in title_list:

                title_movies = self.find_movies_by_title(title)

                # only positive or negative, and all movies are in database.

                movie_id = title_movies[0]

#                 tuples.append( (title, movie_id, sentiment_val) )
                tuples.append( (title, sentiment_val) )
    
                

        
        return tuples

    def find_movies_closest_to_title(self, title, max_distance=3):
        """Creative Feature: Given a potentially misspelled movie title,
        return a list of the movies in the dataset whose titles have the least
        edit distance from the provided title, and with edit distance at most
        max_distance.

        - If no movies have titles within max_distance of the provided title,
        return an empty list.
        - Otherwise, if there's a movie closer in edit distance to the given
        title than all other movies, return a 1-element list containing its
        index.
        - If there is a tie for closest movie, return a list with the indices
        of all movies tying for minimum edit distance to the given movie.

        Example:
          # should return [1656]
          chatbot.find_movies_closest_to_title("Sleeping Beaty")

        :param title: a potentially misspelled title
        :param max_distance: the maximum edit distance to search for
        :returns: a list of movie indices with titles closest to the given title
        and within edit distance max_distance
        """

        pass

    def disambiguate(self, clarification, candidates):
        """Creative Feature: Given a list of movies that the user could be
        talking about (represented as indices), and a string given by the user
        as clarification (eg. in response to your bot saying "Which movie did
        you mean: Titanic (1953) or Titanic (1997)?"), use the clarification to
        narrow down the list and return a smaller list of candidates (hopefully
        just 1!)

        - If the clarification uniquely identifies one of the movies, this
        should return a 1-element list with the index of that movie.
        - If it's unclear which movie the user means by the clarification, it
        should return a list with the indices it could be referring to (to
        continue the disambiguation dialogue).

        Example:
          chatbot.disambiguate("1997", [1359, 2716]) should return [1359]

        :param clarification: user input intended to disambiguate between the
        given movies
        :param candidates: a list of movie indices
        :returns: a list of indices corresponding to the movies identified by
        the clarification
        """
        pass

    ############################################################################
    # 3. Movie Recommendation helper functions                                 #
    ############################################################################

    @staticmethod
    def binarize(ratings, threshold=2.5):
        """Return a binarized version of the given matrix.

        To binarize a matrix, replace all entries above the threshold with 1.
        and replace all entries at or below the threshold with a -1.

        Entries whose values are 0 represent null values and should remain at 0.

        Note that this method is intentionally made static, as you shouldn't use
        any attributes of Chatbot like self.ratings in this method.

        :param ratings: a (num_movies x num_users) matrix of user ratings, from
         0.5 to 5.0
        :param threshold: Numerical rating above which ratings are considered
        positive

        :returns: a binarized version of the movie-rating matrix
        """
        ########################################################################
        # Binarize the supplied ratings matrix.                                #
        #                                                                      #
        # WARNING: Do not use self.ratings directly in this function.          #
        ########################################################################

        # The starter code returns a new matrix shaped like ratings but full of
        # zeros.
        binarized_ratings = np.copy(ratings)
        for i in range(len(ratings)):
            vec = ratings[i]
            for j in range(len(vec)):
                if vec[j] == 0:
                    continue
                elif vec[j] > threshold:
                    binarized_ratings[i][j] = 1
                elif vec[j] <= threshold:
                    binarized_ratings[i][j] = -1
        
        return binarized_ratings

        ########################################################################
        #                        END OF YOUR CODE                              #
        ########################################################################
        return binarized_ratings

    def similarity(self, u, v):
        """Calculate the cosine similarity between two vectors.

        You may assume that the two arguments have the same shape.

        :param u: one vector, as a 1D numpy array
        :param v: another vector, as a 1D numpy array

        :returns: the cosine similarity between the two vectors
        """
        ########################################################################
        # Compute cosine similarity between the two vectors.             #
        ########################################################################
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        if np.any(u) and np.any(v):
            return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
        return 0

    def recommend(self, user_ratings, ratings_matrix, k=10, creative=False):
        """Generate a list of indices of movies to recommend using collaborative
         filtering.

        You should return a collection of `k` indices of movies recommendations.

        As a precondition, user_ratings and ratings_matrix are both binarized.

        Remember to exclude movies the user has already rated!

        Please do not use self.ratings directly in this method.

        :param user_ratings: a binarized 1D numpy array of the user's movie
            ratings
        :param ratings_matrix: a binarized 2D numpy matrix of all ratings, where
          `ratings_matrix[i, j]` is the rating for movie i by user j
        :param k: the number of recommendations to generate
        :param creative: whether the chatbot is in creative mode

        :returns: a list of k movie indices corresponding to movies in
        ratings_matrix, in descending order of recommendation.
        """

        ########################################################################
        # Implement a recommendation function that takes a vector        #
        # user_ratings and matrix ratings_matrix and outputs a list of movies  #
        # recommended by the chatbot.                                          #
        #                                                                      #
        # WARNING: Do not use the self.ratings matrix directly in this         #
        # function.                                                            #
        #                                                                      #
        # For starter mode, you should use item-item collaborative filtering   #
        # with cosine similarity, no mean-centering, and no normalization of   #
        # scores.                                                              #
        ########################################################################

        # Populate this list with k movie indices to recommend to the user.
        recommendations = []
        ratings_predictions = {}
        
        # Find a movie the user has not rated yet.
        for i in range(len(user_ratings)):
            predicted_rating = 0
            rating = user_ratings[i]
            if rating == 0:
                # Compare movie i to each movie the user has already rated. 
                for j in range(len(user_ratings)):
                    if user_ratings[j] != 0 and i != j:
                        predicted_rating += self.similarity(ratings_matrix[i], ratings_matrix[j]) * user_ratings[j]

                ratings_predictions[i] = predicted_rating
        
        for key in sorted(ratings_predictions, key=ratings_predictions.get, reverse=True)[:k]:
            recommendations.append(key)

        ########################################################################
        #                        END OF YOUR CODE                              #
        ########################################################################
        return recommendations

    ############################################################################
    # 4. Debug info                                                            #
    ############################################################################

    def debug(self, line):
        """
        Return debug information as a string for the line string from the REPL

        NOTE: Pass the debug information that you may think is important for
        your evaluators.
        """
        debug_info = 'debug info'
        return debug_info

    ############################################################################
    # 5. Write a description for your chatbot here!                            #
    ############################################################################
    def intro(self):
        """Return a string to use as your chatbot's description for the user.

        Consider adding to this description any information about what your
        chatbot can do and how the user can interact with it.
        """
        return """
        Your task is to implement the chatbot as detailed in the PA6
        instructions.
        Remember: in the starter mode, movie names will come in quotation marks
        and expressions of sentiment will be simple!
        TODO: Write here the description for your own chatbot!
        """


if __name__ == '__main__':
    print('To run your chatbot in an interactive loop from the command line, '
          'run:')
    print('    python3 repl.py')
