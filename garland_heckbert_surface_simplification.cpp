#include <algorithm>
#include <functional>
#include <iostream>
#include <limits>
#include <queue>
#include <set>
#include <utility>
#include <unordered_map>
#include <vector>
#include <Eigen/Core>
#include <igl/bfs_orient.h>
#include <igl/cat.h>
#include <igl/collapse_edge.h>
#include <igl/doublearea.h>
#include <igl/edge_flaps.h>
#include <igl/is_edge_manifold.h>
#include <igl/knn.h>
#include <igl/list_to_matrix.h>
#include <igl/octree.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/per_face_normals.h>
#include <igl/read_triangle_mesh.h>
#include <igl/remove_duplicate_vertices.h>
#include <igl/remove_unreferenced.h>
#include <igl/slice_mask.h>
#include <igl/unique_rows.h>
#include <igl/vertex_triangle_adjacency.h>
#include <igl/writeOBJ.h>

int main(int argc, char* argv[]){
    Eigen::MatrixXd vertex, vertex_, original_vertex;
    Eigen::MatrixXi face, face_, original_face;
    if(!igl::read_triangle_mesh(argv[1], vertex, face)){
        std::cerr << "Error Loading Mesh" << std::endl;
        return 0;
    }
    original_face = face;
    original_vertex = vertex;
    Eigen::VectorXi index_map, reverse_map;
    igl::remove_duplicate_vertices(vertex, face, 1e-7, vertex_, index_map, reverse_map, face_);
    Eigen::Array<bool, Eigen::Dynamic, 1> degenerate_mask = (face_.col(0).array() != face_.col(1).array()) && (face_.col(1).array() != face_.col(2).array()) && (face_.col(2).array() != face_.col(0).array());
    Eigen::MatrixXi unique_face;
    igl::slice_mask(face_, degenerate_mask, 1, unique_face);
    Eigen::VectorXd double_area;
    igl::doublearea(vertex_, unique_face, double_area);
    degenerate_mask = double_area.array() > 1e-12;
    igl::slice_mask(unique_face, degenerate_mask, 1, face_);
    igl::remove_unreferenced(vertex_, face_, vertex, face, index_map);
    Eigen::VectorXi face_component;
    igl::bfs_orient(face, face, face_component);
    bool manifold = igl::is_edge_manifold(face);
    Eigen::MatrixXd normal;
    igl::per_face_normals(vertex, face, normal);
    std::vector<Eigen::Matrix4d> quadric(vertex.rows(), Eigen::Matrix4d::Zero());
    for(int _ = 0; _ < face.rows(); ++_){
        Eigen::Vector3d unit_vector = normal.row(_);
        Eigen::Vector4d plane(unit_vector(0), unit_vector(1), unit_vector(2), -unit_vector.dot(vertex.row(face(_, 0))));
        Eigen::Matrix4d error_quadric = plane * plane.transpose();
        for(int j = 0; j < 3; ++j) quadric[face(_, j)] += (error_quadric * (double_area(_) * 0.5));
    }
    int initial_face = face.rows();
    Eigen::MatrixXi edge, edge_flap, edge_index;
    Eigen::VectorXi edge_map;
    igl::edge_flaps(face, edge, edge_map, edge_flap, edge_index);
    double penalty_factor = 1e3;
    for(int _ = 0;  _ < edge.rows(); ++_){
        if(edge_flap(_, 0) == -1 || edge_flap(_, 1) == -1){
            Eigen::RowVector3d edge_vector = (vertex.row(edge(_, 1)) - vertex.row(edge(_, 0))).normalized();
            int index = (edge_flap(_, 0) != -1) ? edge_flap(_, 0) : edge_flap(_, 1);
            Eigen::RowVector3d unit_vector = normal.row(index);
            Eigen::RowVector3d perpendicular_plane = edge_vector.cross(unit_vector).normalized();
            Eigen::Vector4d constraint_plane(perpendicular_plane(0), perpendicular_plane(1), perpendicular_plane(2), -perpendicular_plane.dot(vertex.row(edge(_, 0))));
            Eigen::Matrix4d error_quadric = penalty_factor * (constraint_plane * constraint_plane.transpose());
            quadric[edge(_, 0)] += error_quadric;
            quadric[edge(_, 1)] += error_quadric;
        }
    }
    std::set<std::pair<int, int>> edge_set;
    for(int _ = 0; _ < edge.rows(); ++_) edge_set.insert({std::min(edge(_, 0), edge(_, 1)), std::max(edge(_, 0), edge(_, 1))});
    std::vector<std::vector<int>> point_index;
    Eigen::MatrixXi children;
    Eigen::MatrixXd center_coordinate;
    Eigen::VectorXd width;
    igl::octree(vertex, point_index, children, center_coordinate, width);
    short neighbour = 10;
    Eigen::MatrixXi I;
    igl::knn(vertex, neighbour, point_index, children, center_coordinate, width, I);
    std::vector<std::vector<int>> threshold_edge;
    double distance_threshold = (vertex.colwise().maxCoeff() - vertex.colwise().minCoeff()).norm() * 0.01;
    double distance_threshold_squared = distance_threshold * distance_threshold;
    for(int _ = 0; _ < I.rows(); ++_){
        for(int j = 0; j < I.cols(); j++){
            if(_ <  I(_, j) && (vertex.row(_) - vertex.row(I(_, j))).squaredNorm() < distance_threshold_squared) threshold_edge.push_back({_, I(_, j)});
        }
    }
    std::sort(threshold_edge.begin(), threshold_edge.end());
    threshold_edge.erase(std::unique(threshold_edge.begin(), threshold_edge.end()), threshold_edge.end());
    std::vector<std::pair<int, int>> unique_threshold;
    for(auto& _ : threshold_edge){
        if(edge_set.find({_[0], _[1]}) == edge_set.end()) unique_threshold.push_back({_[0], _[1]});
    }
    auto optimal_contraction = [&](int v_1, int v_2, double& cost, Eigen::RowVector3d& optimal) -> void {
        if(v_1 < 0 || v_2 < 0 || v_1 >= quadric.size() || v_2 >= quadric.size()){
            cost = std::numeric_limits<double>::max();
            return;
        }
        Eigen::Matrix4d quadric_ = quadric[v_1] + quadric[v_2];
        Eigen::Matrix4d system = quadric_;
        system.row(3) << 0, 0, 0, 1;
        Eigen::Vector4d target(0, 0, 0, 1);
        Eigen::RowVector3d v_1_ = vertex.row(v_1);
        Eigen::RowVector3d v_2_ = vertex.row(v_2);
        Eigen::RowVector3d v_0_ = (vertex.row(v_1) + vertex.row(v_2)) * 0.5; //  A matrix is numerically singular if its smallest pivot is tiny compared to its largest pivot.
        if(std::abs(system.determinant()) > 1e-6 * std::pow(system.norm(), system.rows())){
            Eigen::Matrix3d A = quadric_.block<3,3>(0,0);
            Eigen::Vector3d b = -quadric_.block<3,1>(0,3);
            Eigen::Vector3d diagonal = Eigen::Vector3d::Zero();
            Eigen::Matrix3d lower_triangular = Eigen::Matrix3d::Identity();
            for(int j = 0; j < 3; ++j){
                double d = 0;
                for(int k = 0; k < j; ++k) d += lower_triangular(j,k) * lower_triangular(j,k) * diagonal(k);
                diagonal(j) = A(j, j) - d;
                if(std::abs(diagonal(j)) < 1e-6) goto subset_placement;
                for(int i = j + 1; i < 3; ++i){
                    double l = 0;
                    for(int k = 0; k < j; ++k) l += lower_triangular(i, k) * lower_triangular(j, k) * diagonal(k);
                    lower_triangular(i, j) = (A(i, j) - l) / diagonal(j);
                }
            }
            Eigen::Vector3d intermediate_solution = Eigen::Vector3d::Zero();
            for(int i = 0; i < 3; ++i){
                double _ = 0;
                for(int j = 0; j < i; ++j) _ += lower_triangular(i, j) * intermediate_solution(j);
                intermediate_solution(i) = b(i) - _;
            }
            Eigen::Vector3d scaled_intermediate_solution = intermediate_solution.array() / diagonal.array();
            Eigen::Vector3d solution = Eigen::Vector3d::Zero();
            for(int i = 2; i >= 0; --i){
                double _ = 0;
                for(int j = i + 1; j < 3; ++j) _ += lower_triangular(j, i) * solution(j);
                solution(i) = scaled_intermediate_solution(i) - _;
            }
            //Eigen::VectorXd solution = system.ldlt().solve(target);
            optimal = solution.transpose();
            Eigen::Vector4d x_(solution(0), solution(1), solution(2), 1.0);
            cost = std::max(0.0, (x_.transpose() * quadric_ * x_).value());
        } else{
            subset_placement:
            target.head<3>() = v_1_;
            double e1 = (target.transpose() * quadric_ * target).value();
            target.head<3>() = v_2_;
            double e2 = (target.transpose() * quadric_ * target).value();
            target.head<3>() = v_0_;
            double e0 = (target.transpose() * quadric_ * target).value();
            if(e1 <= e2 && e1 <= e0){
                cost = e1;
                optimal = v_1_;
            } else if(e2 <= e1 && e2 <= e0){
                cost = e2;
                optimal = v_2_;
            } else{
                cost = e0;
                optimal = v_0_;
            }
        }
    };
    struct Edge{
        double cost;
        int v_1, v_2, edge;
        Eigen::RowVector3d optimal;
        int version;
        bool operator>(const Edge& other) const {return cost > other.cost;}
    };
    double cost = 0;
    Eigen::RowVector3d optimal;
    std::priority_queue<Edge, std::vector<Edge>, std::greater<Edge>> candidate_pair;
    Eigen::VectorXi edge_version = Eigen::VectorXi::Zero(edge.rows() + unique_threshold.size());
    for(int _ = 0; _ < edge.rows(); ++_){
        optimal_contraction(edge(_, 0), edge(_, 1), cost, optimal);
        candidate_pair.push({cost, edge(_, 0), edge(_, 1), _, optimal, edge_version[_]});
    }
    for(int _ = 0; _ < unique_threshold.size(); ++_){
        int pos = edge.rows() + _;
        optimal_contraction(unique_threshold[_].first, unique_threshold[_].second, cost, optimal);
        candidate_pair.push({cost, unique_threshold[_].first, unique_threshold[_].second, pos, optimal, edge_version[pos]});
    }
    std::vector<std::vector<int>> vertex_face, vertex_corner;
    igl::vertex_triangle_adjacency(vertex, face, vertex_face, vertex_corner);
    for(auto& _ : unique_threshold){
        vertex_face[_.first].push_back(_.second);
        vertex_face[_.second].push_back(_.first);
    }

    std::vector<bool> active_face(face.rows());
    int target = face.rows() * 0.30, current = face.rows();
    int breakdown = 0;
    auto normal_flip = [&](int v1, int v2, Eigen::RowVector3d optimal) -> bool {
        std::vector<int> adjacent_vertex = {v1, v2};
        for(int v : adjacent_vertex){
            for(int _ : vertex_face[v]){
                Eigen::RowVector3i face_vector = face.row(_);
                if((face_vector[0] == v1 || face_vector[1] == v1 || face_vector[2] == v1) && (face_vector[0] == v1 || face_vector[1] == v1 || face_vector[2] == v1)) continue;
                Eigen::RowVector3d v_0 = vertex.row(face_vector[0]), v_1 = vertex.row(face_vector[1]), v_2 = vertex.row(face_vector[2]);
                Eigen::RowVector3d unit_vector = (v_1 - v_0).cross(v_2 - v_0).normalized();
                if(face_vector[0] == v) v_0 = optimal;
                if(face_vector[1] == v) v_1 = optimal;
                if(face_vector[2] == v) v_2 = optimal;
                Eigen::RowVector3d unit_vector_ = (v_1 - v_0).cross(v_2 - v_0);
                if(unit_vector_.norm() < 1e-12) return 1;
                unit_vector_.normalize();
                if(unit_vector.dot(unit_vector_) < 0.1) return 1;
            }
        }
        return 0;
    };
    while(current > target && !candidate_pair.empty()){
        Edge top = candidate_pair.top();
        candidate_pair.pop();
        //if(!igl::is_edge_manifold(face)) continue;
        if(top.cost > 1e10) break;
        if(top.version != edge_version[top.edge]) continue;
        if(edge_flap(top.edge, 0) == -1 && edge_flap(top.edge, 1) == -1) continue;
        int vA, vB;
        if(top.edge < edge.rows()){
            vA = edge(top.edge, 0);
            vB = edge(top.edge, 1);
        } else{ 
            vA = top.v_1;
            vB = top.v_2;
            Eigen::RowVector3d optimal = top.optimal;
            if(normal_flip(vA, vB, optimal)){
                top.cost *= 100;
                candidate_pair.push(top);
                continue;
            }
            vertex.row(vA) = optimal;
            quadric[vA] += quadric[vB];
            edge_version[top.edge]++;
            for(int _ : vertex_face[vB]){
                if(_ == -1) continue;
                for(int i = 0; i < 3; ++i) if(face(_, i) == vB) face(_, i) = vA;
                short frequency = 0;
                for(int i = 0; i < 3; ++i) if(face(_, i) == vA) frequency++;
                if(frequency >= 2) face.row(_).setConstant(-1);
                else vertex_face[vA].push_back(_);
            }
            for(int _ : vertex_face[vB]){
                for(int i = 0; i < 3; ++i){
                    if(face(_, i) != vA  && face(_, i) == -1){
                        optimal_contraction(vA, face(_, i), cost, optimal);
                        candidate_pair.push({cost, vA, face(_, i), top.edge, optimal, edge_version[top.edge]});
                    }
                }
            }
            vertex_face[vB].clear();
            continue;
        }
        if(vA == -1 || vB == -1) continue;
        std::set<int> neighbour_a, neighbour_b;
        for(int f : vertex_face[vA]){
            if(f == -1 || face(f, 0) == -1) continue;
            for(int i = 0; i < 3; ++i){
                int id = edge_map[f + i * initial_face];
                if(id != -1) edge_version[id]++;
                if(face(f,i) != vA) neighbour_a.insert(face(f,i));
            }
        }
        for(int f : vertex_face[vB]){
            if(f == -1 || face(f, 0) == -1) continue;
            for(int i = 0; i < 3; ++i){
                int id = edge_map[f + i * initial_face];
                if(id != -1) edge_version[id]++;
                if(face(f,i) != vB) neighbour_b.insert(face(f,i));
            }
        }
        int neighbor = 0; // If the two vertices of the edge share more than 2 common neighbors, collapsing them will create a non-manifold pinch.
        for(int fA : neighbour_a){
            if(neighbour_b.count(fA)) neighbor++;
        }
        if(neighbor > 2) continue;
        int v1, v2, f1, f2;
        if(igl::collapse_edge(top.edge, top.optimal, vertex, face, edge, edge_map, edge_flap, edge_index, v1, v2, f1, f2)){
            if(v1 == -1 && v2 == -1) continue;
            breakdown++;
            if(f1 != -1){
                current--;
                active_face[f1] = 0;
                face.row(f1).setConstant(-1);
            }
            if(f2 != -1){
                current--;
                active_face[f2] = 0;
                face.row(f2).setConstant(-1);
            }
            int max_v = std::max(v1, v2);
            if(max_v >= quadric.size()) quadric.resize(max_v + 1, Eigen::Matrix4d::Zero());
            if(max_v >= vertex_face.size()) vertex_face.resize(max_v + 1);
            quadric[v1] += quadric[v2];
            //vertex_face[v1].insert(vertex_face[v1].end(), vertex_face[v2].begin(), vertex_face[v2].end());
            for(int f : vertex_face[v2]){
                if(f != -1 && face(f, 0) != -1) vertex_face[v1].push_back(f);
            }
            vertex_face[v2].clear();
            for(int f : vertex_face[v1]){
                if(f < 0 || f >= face.rows() || face(f,0) == -1) continue;
                for(int i = 0; i < 3; ++i){
                    int map_idx = f + i * initial_face;
                    if (map_idx >= 0 && map_idx < edge_map.size()){
                        int e_id = edge_map[map_idx];
                        if(e_id != -1 && e_id < edge_version.size()){
                            edge_version[e_id]++;
                            double cost;
                            Eigen::RowVector3d optimal;
                            optimal_contraction(edge(e_id, 0), edge(e_id, 1), cost, optimal);
                            candidate_pair.push({cost, edge(e_id, 0), edge(e_id, 1), e_id, optimal, edge_version[e_id]});
                        }
                    }
                }
            }
        } else{
            //std::cout << "False" <<std::endl;
        }
    }
    
    Eigen::MatrixXi condensed_faces(current, 3);
    int row_idx = 0;
    for(int i = 0; i < face.rows(); ++i) {
        if(active_face[i] && face(i,0) != -1) {
            condensed_faces.row(row_idx++) = face.row(i);
        }
    }
    //face = condensed_faces;
    /*int valid_face = 0;
    Eigen::MatrixXi updated_face(face.rows(), 3);
    for(int _ = 0; _ < face.rows(); ++_){
        if(face(_,0) != -1 && face(_,1) != -1 && face(_,2) != -1) updated_face.row(valid_face++) = face.row(_);
    }
    updated_face.conservativeResize(valid_face, 3);*/
    std::vector<Eigen::RowVector3i> valid_list;
    for(int i = 0; i < face.rows(); i++){
        if(face(i,0) != -1 && face(i,1) != -1 && face(i,2) != -1) {
            valid_list.push_back(face.row(i));
        }
    }
    Eigen::MatrixXi updated_face(valid_list.size(), 3);
    for(int i = 0; i < valid_list.size(); i++) updated_face.row(i) = valid_list[i];
    Eigen::VectorXi vertex_reference;
    Eigen::VectorXd face_reference;
    igl::remove_unreferenced(vertex, updated_face, vertex_, face_, index_map, face_reference);
    //igl::remove_unreferenced(vertex_, face_, vertex_, face_, vertex_reference, face_reference);
    igl::writeOBJ("result.obj", vertex_, face_);
    std::cout << "Original Mesh: " << original_vertex.rows() << " vertices, " << original_face.rows() << " faces." << std::endl;
    std::cout << "Simplified Mesh: " << vertex_.rows() << " vertices, " << face_.rows() << " faces." << std::endl;
    igl::opengl::glfw::Viewer viewer;
    int input = viewer.data().id;
    viewer.data(input).set_mesh(original_vertex, original_face);
    viewer.data(input).set_colors(Eigen::RowVector3d(0.9, 0.9, 0.9));
    viewer.data(input).show_lines = 1; 
    viewer.data(input).line_color = Eigen::RowVector4f(1, 0, 0, 1);
    viewer.data(input).compute_normals();
    viewer.data(input).face_based = 1;
    int output = viewer.append_mesh();
    viewer.data(output).clear();
    viewer.data(output).set_mesh(vertex_, face_);
    viewer.data(output).set_colors(Eigen::RowVector3d(0.9, 0.9, 0.9));
    viewer.data(output).show_lines = 1;
    viewer.data(output).line_color = Eigen::RowVector4f(0, 1, 0, 1);
    viewer.data(output).compute_normals();
    viewer.data(output).face_based = 1;
    viewer.callback_init = [&](igl::opengl::glfw::Viewer& view){
        view.core().viewport = Eigen::Vector4f(0, 0, 480, 640);
        view.append_core(Eigen::Vector4f(480, 0, 480, 640));
        int left = view.core(0).id, right = view.core(1).id;
        for(auto& data : view.data_list){
            data.set_visible(0, left);
            data.set_visible(0, right);
        }
        view.data(input).set_visible(1, left);
        view.core(0).align_camera_center(view.data(input).V, view.data(input).F);
        view.data(output).set_visible(1, right);
        view.core(1).align_camera_center(view.data(output).V, view.data(output).F);
        view.core(1).light_position = view.core(0).light_position;
        return 0;
    };
    viewer.callback_post_draw = [&](igl::opengl::glfw::Viewer& view){
        view.core(1).camera_zoom = view.core(0).camera_zoom;
        view.core(1).trackball_angle = view.core(0).trackball_angle;
        view.core(1).camera_translation = view.core(0).camera_translation;
        return 0;
    };
    viewer.callback_post_resize = [&](igl::opengl::glfw::Viewer &view, int w, int h){
        view.core(0).viewport = Eigen::Vector4f(0, 0, w / 2, h);
        view.core(1).viewport = Eigen::Vector4f(w / 2, 0, w / 2, h);
        return 0;
    };
    viewer.launch();
    return 1;
}
//c_cpp -file garland_heckbert_surface_simplification.cpp, glad.c -link libigl -flag "-O2","-Wa,-mbig-obj","-Wl,--stack,16777216" -args torus.obj