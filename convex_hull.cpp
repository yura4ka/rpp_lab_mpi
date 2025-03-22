#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <mpi.h>

struct Point
{
    double x, y;

    Point() : x(0), y(0) {}
    Point(double x_val, double y_val) : x(x_val), y(y_val) {}

    double angle(const Point& p1) const
    {
        return std::atan2(p1.y - this->y, p1.x - this->x);
    }

    double distance(const Point& p1) const
    {
        return std::pow(p1.x - this->x, 2) + std::pow(p1.y - this->y, 2);
    }
};

int orientation(const Point& p1, const Point& p2, const Point& p3)
{
    double val = (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x);
    if (val == 0) return 0;
    if (val > 0) return 1;
    return -1;
}

std::vector<Point> convex_hull(std::vector<Point>& points) {
    if (points.size() <= 3) return points;

    Point start = *std::min_element(points.begin(), points.end(), [](const Point& p1, const Point& p2) {
        return p1.y < p2.y || (p1.y == p2.y && p1.x < p2.x);
        });

    std::sort(points.begin(), points.end(), [&start](const Point& p1, const Point& p2) {
        return start.angle(p1) < start.angle(p2) || (start.angle(p1) == start.angle(p2) && start.distance(p1) < start.distance(p2));
        });

    std::vector<Point> hull = { points[0], points[1] };

    for (int i = 2; i < points.size(); ++i)
    {
        while (hull.size() > 1 && orientation(hull[hull.size() - 2], hull[hull.size() - 1], points[i]) <= 0) {
            hull.pop_back();
        }
        hull.push_back(points[i]);
    }

    return hull;
}

void input_points(const char* filename, std::vector<Point>& points)
{
    int n;

    std::ifstream file(filename);
    file >> n;

    for (int i = 0; i < n; ++i)
    {
        double x, y;
        file >> x >> y;
        points.push_back(Point(x, y));
    }
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Datatype MPI_POINT;
    MPI_Type_contiguous(2, MPI_DOUBLE, &MPI_POINT);
    MPI_Type_commit(&MPI_POINT);

    std::vector<Point> points;

    if (rank == 0) {
        input_points(argv[1], points);
    }

    double time_start = MPI_Wtime();
    int n = points.size();
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int points_per_proc = n / size;
    int remainder = n % size;

    std::vector<int> sendcounts(size, points_per_proc);
    std::vector<int> displacements(size, 0);
    for (int i = 0; i < remainder; ++i) {
        sendcounts[i]++;
    }
    for (int i = 1; i < size; ++i) {
        displacements[i] = displacements[i - 1] + sendcounts[i - 1];
    }

    std::vector<Point> local_points(sendcounts[rank]);
    MPI_Scatterv(&points[0], &sendcounts[0], &displacements[0], MPI_POINT,
        &local_points[0], sendcounts[rank], MPI_POINT,
        0, MPI_COMM_WORLD);

    auto hull = convex_hull(local_points);

    std::vector<int> all_hull_sizes(size);
    int local_hull_size = hull.size();

    MPI_Gather(&local_hull_size, 1, MPI_INT, &all_hull_sizes[0], 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<Point> global_points;
    if (rank == 0) {
        int total_hull_size = 0;
        for (int i = 0; i < size; ++i) {
            total_hull_size += all_hull_sizes[i];
        }
        global_points.resize(total_hull_size);
    }

    std::vector<int> displacements_for_gather(size, 0);
    for (int i = 1; i < size; ++i) {
        displacements_for_gather[i] = displacements_for_gather[i - 1] + all_hull_sizes[i - 1];
    }

    MPI_Gatherv(&hull[0], local_hull_size, MPI_POINT,
        &global_points[0], &all_hull_sizes[0], &displacements_for_gather[0], MPI_POINT,
        0, MPI_COMM_WORLD);

    if (rank == 0) {
        auto result = convex_hull(global_points);
        double time_end = MPI_Wtime();
        std::cout << "Input size: " << points.size()
            << "\nNumber of processes: " << size
            << "\nTime: " << time_end - time_start
            << "\nHull size: " << hull.size() << "\n";
    }

    MPI_Finalize();
    return 0;
}