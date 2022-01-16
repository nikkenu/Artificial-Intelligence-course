import csv
import sys

from util import Node, QueueFrontier, StackFrontier

# Maps names to a set of corresponding person_ids
people_to_ids = {}

# Maps person_ids to a dictionary of: name, birth, movies (a set of movie_ids)
people = {}

# Maps movie_ids to a dictionary of: title, year, stars (a set of person_ids)
movies = {}


def load_data(directory, print_messages):
    """
    Load data from CSV files into memory
    """

    if print_messages:
        print(f"Loading data from '{directory}' ...")

    # Load people and construct people_to_ids mapping
    with open(f"{directory}/people.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            people[row["id"]] = {
                "name": row["name"],
                "birth": row["birth"],
                "movies": set()
            }
            if row["name"].lower() not in people_to_ids:
                people_to_ids[row["name"].lower()] = {row["id"]}
            else:
                people_to_ids[row["name"].lower()].add(row["id"])

    # Load movies
    with open(f"{directory}/movies.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            movies[row["id"]] = {
                "title": row["title"],
                "year": row["year"],
                "stars": set()
            }

    # Load stars and link them to people and movies
    with open(f"{directory}/stars.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                people[row["person_id"]]["movies"].add(row["movie_id"])
                movies[row["movie_id"]]["stars"].add(row["person_id"])
            except KeyError:
                pass

    if print_messages:
        print("Data loaded.")


def main(directory, is_verbose=True):

    # Load data from files into memory
    load_data(directory, is_verbose)

    name_prompt = "Name: " if is_verbose else ""
    continue_prompt = "Try again (Y/N)? " if is_verbose else ""

    while True:

        # Ask user for names
        star_name_1 = input(name_prompt)
        star_name_2 = input(name_prompt)

        # Find ID for the first star
        source = person_id_for_name(star_name_1, is_verbose)
        if source is None:
            sys.exit("Person 1 not found.")

        # Find ID for the second star
        target = person_id_for_name(star_name_2, is_verbose)
        if target is None:
            sys.exit("Person 2 not found.")

        # Find shortest path between source and target stars
        path = shortest_path(source, target)

        if path is None:
            print("Not connected.")
        else:
            degrees = len(path)
            print(f"{degrees} degrees of separation.")
            path = [(None, source)] + path
            if is_verbose:
                for i in range(degrees):
                    person1 = people[path[i][1]]["name"]
                    person2 = people[path[i + 1][1]]["name"]
                    movie = movies[path[i + 1][0]]["title"]
                    print(f"{i + 1}: {person1} and {person2} starred in {movie}")

        give_another_pair = input(continue_prompt)
        if (give_another_pair.upper() != "Y"):
            break


def shortest_path(source, target):
    """
    Returns the shortest list of (movie_id, person_id) pairs
    that connect the source to the target.

    If no possible path, returns None.
    """
    discovered = []
    queue = QueueFrontier()
    queue.add(Node(source, None, people[source]["movies"]))

    while not queue.empty():
        tmp_node = queue.remove()

        if tmp_node.state == target:
            return path_returner(tmp_node)

        for movie_id, person_id in neighbors_for_person(tmp_node.state):
            if person_id not in discovered:
                n_node = Node(person_id, tmp_node, people[person_id]["movies"])
                queue.add(n_node)
                discovered.append(person_id)

    return None

def path_returner(node):
    path = []
    while node.parent is not None:
        for movie in node.action:
            if movie in node.parent.action:
                mutual_movie = movie

        path.insert(0, (mutual_movie, node.state))
        node = node.parent
    return path


def person_id_for_name(name, is_verbose):
    """
    Returns the IMDB id for a person's name,
    resolving ambiguities as needed.
    """
    person_ids = list(people_to_ids.get(name.lower(), set()))
    if len(person_ids) == 0:
        return None
    elif len(person_ids) > 1:
        if is_verbose:
            print(f"Which '{name}'?")

            for person_id in person_ids:
                person = people[person_id]
                name = person["name"]
                birth = person["birth"]
                print(f"ID: {person_id}, Name: {name}, Birth: {birth}")
        try:
            id_prompt = "Intended Person ID: " if is_verbose else ""
            person_id = input(id_prompt)
            if person_id in person_ids:
                return person_id
        except ValueError:
            pass
        return None
    else:
        return person_ids[0]


def neighbors_for_person(person_id):
    """
    Returns (movie_id, person_id) pairs for people
    who starred with a given person.
    """

    movie_ids = people[person_id]["movies"]
    neighbors = set()
    for movie_id in movie_ids:
        for person_id in movies[movie_id]["stars"]:
            neighbors.add((movie_id, person_id))

    return neighbors


if __name__ == "__main__":
    verbose = True
    directory = None

    if len(sys.argv) == 3 and sys.argv[1] == "--quiet":
        # disable user prompts and other extra output (used inside grader)
        verbose = False
        directory = sys.argv[2]
    elif len(sys.argv) == 2:
        if sys.argv[1] == "--quiet":
            # disable user prompts and other extra output (used inside grader)
            verbose = False
            # use large dataset by default if directory is not provided
            directory = "large"
        else:
            directory = sys.argv[1]
    elif len(sys.argv) == 1:
        # use large dataset by default if directory is not provided
        directory = "large"

    if directory is None or len(directory.strip()) == 0:
        sys.exit("Usage: python degrees.py [--quiet] [directory]")

    main(directory, verbose)
