#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include <hpp/fcl/collision_data.h>
#include <hpp/fcl/distance.h>
#include <hpp/fcl/shape/geometric_shapes.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {

constexpr long long kKernelAbiVersion = 3;

double dot3(const double* a, const double* b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

double clamp01(double value) {
    return std::max(0.0, std::min(1.0, value));
}

void closest_segment_points(
    const double* first_start,
    const double* first_end,
    const double* second_start,
    const double* second_end,
    double* first_point,
    double* second_point
) {
    double first_axis[3];
    double second_axis[3];
    double offset[3];
    for (int axis = 0; axis < 3; ++axis) {
        first_axis[axis] = first_end[axis] - first_start[axis];
        second_axis[axis] = second_end[axis] - second_start[axis];
        offset[axis] = first_start[axis] - second_start[axis];
    }
    const double first_length_squared = dot3(first_axis, first_axis);
    const double second_length_squared = dot3(second_axis, second_axis);
    const double first_offset = dot3(first_axis, offset);
    const double second_offset = dot3(second_axis, offset);
    constexpr double epsilon = 1e-15;
    double first_parameter = 0.0;
    double second_parameter = 0.0;
    if (first_length_squared <= epsilon && second_length_squared <= epsilon) {
        first_parameter = 0.0;
        second_parameter = 0.0;
    } else if (first_length_squared <= epsilon) {
        second_parameter = clamp01(second_offset / second_length_squared);
    } else {
        if (second_length_squared <= epsilon) {
            first_parameter = clamp01(-first_offset / first_length_squared);
        } else {
            const double axes_dot = dot3(first_axis, second_axis);
            const double denominator =
                first_length_squared * second_length_squared - axes_dot * axes_dot;
            if (denominator > epsilon) {
                first_parameter = clamp01(
                    (axes_dot * second_offset - first_offset * second_length_squared)
                    / denominator
                );
            }
            second_parameter =
                (axes_dot * first_parameter + second_offset) / second_length_squared;
            if (second_parameter < 0.0) {
                second_parameter = 0.0;
                first_parameter = clamp01(-first_offset / first_length_squared);
            } else if (second_parameter > 1.0) {
                second_parameter = 1.0;
                first_parameter = clamp01(
                    (axes_dot - first_offset) / first_length_squared
                );
            }
        }
    }
    for (int axis = 0; axis < 3; ++axis) {
        first_point[axis] = first_start[axis] + first_parameter * first_axis[axis];
        second_point[axis] = second_start[axis] + second_parameter * second_axis[axis];
    }
}

py::tuple segment_union_distances(
    py::array_t<long long, py::array::c_style | py::array::forcecast> pair_indices,
    py::array_t<long long, py::array::c_style | py::array::forcecast> first_indices,
    py::array_t<long long, py::array::c_style | py::array::forcecast> second_indices,
    py::array_t<double, py::array::c_style | py::array::forcecast> starts,
    py::array_t<double, py::array::c_style | py::array::forcecast> ends,
    py::array_t<double, py::array::c_style | py::array::forcecast> radii
) {
    const auto pairs = pair_indices.unchecked<1>();
    const auto first = first_indices.unchecked<1>();
    const auto second = second_indices.unchecked<1>();
    const auto segment_starts = starts.unchecked<2>();
    const auto segment_ends = ends.unchecked<2>();
    const auto segment_radii = radii.unchecked<1>();
    if (
        pairs.shape(0) != first.shape(0)
        || pairs.shape(0) != second.shape(0)
        || segment_starts.shape(1) != 3
        || segment_ends.shape(1) != 3
        || segment_starts.shape(0) != segment_ends.shape(0)
        || segment_starts.shape(0) != segment_radii.shape(0)
    ) {
        throw std::invalid_argument("incompatible segment-union kernel inputs");
    }
    std::vector<long long> output_pairs;
    std::vector<long long> output_combinations;
    std::vector<double> output_distances;
    std::vector<double> output_first_points;
    std::vector<double> output_second_points;
    long long current_pair = std::numeric_limits<long long>::min();
    for (py::ssize_t combination = 0; combination < pairs.shape(0); ++combination) {
        const long long pair = pairs(combination);
        const long long first_index = first(combination);
        const long long second_index = second(combination);
        if (
            first_index < 0 || second_index < 0
            || first_index >= segment_starts.shape(0)
            || second_index >= segment_starts.shape(0)
        ) {
            throw std::invalid_argument("segment index is outside the geometry arrays");
        }
        double first_start[3];
        double first_end[3];
        double second_start[3];
        double second_end[3];
        for (int axis = 0; axis < 3; ++axis) {
            first_start[axis] = segment_starts(first_index, axis);
            first_end[axis] = segment_ends(first_index, axis);
            second_start[axis] = segment_starts(second_index, axis);
            second_end[axis] = segment_ends(second_index, axis);
        }
        double first_point[3];
        double second_point[3];
        closest_segment_points(
            first_start,
            first_end,
            second_start,
            second_end,
            first_point,
            second_point
        );
        double squared_distance = 0.0;
        for (int axis = 0; axis < 3; ++axis) {
            const double delta = first_point[axis] - second_point[axis];
            squared_distance += delta * delta;
        }
        const double distance = std::sqrt(squared_distance)
            - segment_radii(first_index) - segment_radii(second_index);
        if (pair != current_pair) {
            if (!output_pairs.empty() && pair < current_pair) {
                throw std::invalid_argument("pair indices must be grouped and ordered");
            }
            current_pair = pair;
            output_pairs.push_back(pair);
            output_combinations.push_back(combination);
            output_distances.push_back(distance);
            output_first_points.insert(output_first_points.end(), first_point, first_point + 3);
            output_second_points.insert(output_second_points.end(), second_point, second_point + 3);
        } else if (distance < output_distances.back()) {
            output_combinations.back() = combination;
            output_distances.back() = distance;
            std::copy(first_point, first_point + 3, output_first_points.end() - 3);
            std::copy(second_point, second_point + 3, output_second_points.end() - 3);
        }
    }
    const py::ssize_t count = static_cast<py::ssize_t>(output_pairs.size());
    py::array_t<long long> result_pairs(count);
    py::array_t<long long> result_combinations(count);
    py::array_t<double> result_distances(count);
    py::array_t<double> result_first_points({count, py::ssize_t(3)});
    py::array_t<double> result_second_points({count, py::ssize_t(3)});
    py::array_t<double> result_fallback_normals({count, py::ssize_t(3)});
    std::copy(output_pairs.begin(), output_pairs.end(), result_pairs.mutable_data());
    std::copy(
        output_combinations.begin(), output_combinations.end(), result_combinations.mutable_data()
    );
    std::copy(output_distances.begin(), output_distances.end(), result_distances.mutable_data());
    std::copy(
        output_first_points.begin(), output_first_points.end(), result_first_points.mutable_data()
    );
    std::copy(
        output_second_points.begin(), output_second_points.end(), result_second_points.mutable_data()
    );
    auto first_witnesses = result_first_points.mutable_unchecked<2>();
    auto second_witnesses = result_second_points.mutable_unchecked<2>();
    auto fallback_normals = result_fallback_normals.mutable_unchecked<2>();
    for (py::ssize_t output = 0; output < count; ++output) {
        const auto combination = output_combinations[output];
        const auto first_index = first(combination);
        const auto second_index = second(combination);
        double delta[3];
        double squared_norm = 0.0;
        for (int axis = 0; axis < 3; ++axis) {
            delta[axis] = first_witnesses(output, axis)
                - second_witnesses(output, axis);
            squared_norm += delta[axis] * delta[axis];
        }
        if (squared_norm <= 1e-24) {
            squared_norm = 0.0;
            for (int axis = 0; axis < 3; ++axis) {
                delta[axis] = 0.5 * (
                    segment_starts(first_index, axis)
                    + segment_ends(first_index, axis)
                    - segment_starts(second_index, axis)
                    - segment_ends(second_index, axis)
                );
                squared_norm += delta[axis] * delta[axis];
            }
        }
        const double norm = std::sqrt(squared_norm);
        for (int axis = 0; axis < 3; ++axis) {
            const double normal = norm > 1e-12
                ? delta[axis] / norm
                : (axis == 0 ? 1.0 : 0.0);
            fallback_normals(output, axis) = delta[axis];
            first_witnesses(output, axis) -= normal * segment_radii(first_index);
            second_witnesses(output, axis) += normal * segment_radii(second_index);
        }
    }
    return py::make_tuple(
        result_pairs,
        result_combinations,
        result_distances,
        result_first_points,
        result_second_points,
        result_fallback_normals
    );
}

class NativeFclUnionKernel {
 public:
    NativeFclUnionKernel(
        py::array_t<long long, py::array::c_style | py::array::forcecast> kinds,
        py::array_t<double, py::array::c_style | py::array::forcecast> dimensions,
        py::array_t<long long, py::array::c_style | py::array::forcecast> pair_indices,
        py::array_t<long long, py::array::c_style | py::array::forcecast> first_indices,
        py::array_t<long long, py::array::c_style | py::array::forcecast> second_indices
    ) {
        const auto kind = kinds.unchecked<1>();
        const auto size = dimensions.unchecked<2>();
        const auto pairs = pair_indices.unchecked<1>();
        const auto first = first_indices.unchecked<1>();
        const auto second = second_indices.unchecked<1>();
        if (
            size.shape(0) != kind.shape(0) || size.shape(1) != 3
            || pairs.shape(0) != first.shape(0)
            || pairs.shape(0) != second.shape(0)
        ) {
            throw std::invalid_argument("incompatible native FCL kernel inputs");
        }
        geometries_.reserve(kind.shape(0));
        for (py::ssize_t index = 0; index < kind.shape(0); ++index) {
            switch (kind(index)) {
                case 0:
                    geometries_.push_back(
                        std::make_shared<hpp::fcl::Sphere>(size(index, 0))
                    );
                    break;
                case 1:
                    geometries_.push_back(
                        std::make_shared<hpp::fcl::Capsule>(
                            size(index, 0), size(index, 1)
                        )
                    );
                    break;
                case 2:
                    geometries_.push_back(
                        std::make_shared<hpp::fcl::Ellipsoid>(
                            size(index, 0), size(index, 1), size(index, 2)
                        )
                    );
                    break;
                default:
                    throw std::invalid_argument("unsupported native FCL geometry kind");
            }
        }
        for (py::ssize_t index = 0; index < pairs.shape(0); ++index) {
            pair_indices_.push_back(pairs(index));
            first_indices_.push_back(first(index));
            second_indices_.push_back(second(index));
        }
    }

    py::tuple compute(
        py::array_t<long long, py::array::c_style | py::array::forcecast> active_pairs,
        py::array_t<double, py::array::c_style | py::array::forcecast> rotations,
        py::array_t<double, py::array::c_style | py::array::forcecast> translations
    ) const {
        const auto active = active_pairs.unchecked<1>();
        const auto rotation = rotations.unchecked<3>();
        const auto translation = translations.unchecked<2>();
        if (
            rotation.shape(0) != static_cast<py::ssize_t>(geometries_.size())
            || rotation.shape(1) != 3 || rotation.shape(2) != 3
            || translation.shape(0) != rotation.shape(0)
            || translation.shape(1) != 3
        ) {
            throw std::invalid_argument("invalid native FCL world placements");
        }
        long long maximum_pair = -1;
        for (const auto pair : pair_indices_) maximum_pair = std::max(maximum_pair, pair);
        std::vector<unsigned char> active_mask(maximum_pair + 1, 0);
        for (py::ssize_t index = 0; index < active.shape(0); ++index) {
            if (active(index) < 0 || active(index) > maximum_pair) {
                throw std::invalid_argument("native FCL active pair is outside the kernel");
            }
            active_mask[active(index)] = 1;
        }
        std::vector<hpp::fcl::Transform3f> transforms(geometries_.size());
        for (py::ssize_t geometry = 0; geometry < rotation.shape(0); ++geometry) {
            hpp::fcl::Matrix3f matrix;
            hpp::fcl::Vec3f vector;
            for (int row = 0; row < 3; ++row) {
                vector[row] = translation(geometry, row);
                for (int column = 0; column < 3; ++column) {
                    matrix(row, column) = rotation(geometry, row, column);
                }
            }
            transforms[geometry].setTransform(matrix, vector);
        }

        std::vector<long long> output_pairs;
        std::vector<long long> output_first;
        std::vector<long long> output_second;
        std::vector<double> output_distances;
        std::vector<double> output_first_points;
        std::vector<double> output_second_points;
        std::vector<double> output_fallback_normals;
        hpp::fcl::DistanceRequest request(true);
        long long current_pair = -1;
        long long exact_calls = 0;
        for (std::size_t combination = 0; combination < pair_indices_.size(); ++combination) {
            const auto pair = pair_indices_[combination];
            if (!active_mask[pair]) continue;
            const auto first = first_indices_[combination];
            const auto second = second_indices_[combination];
            hpp::fcl::DistanceResult result;
            const double distance = hpp::fcl::distance(
                geometries_[first].get(), transforms[first],
                geometries_[second].get(), transforms[second],
                request, result
            );
            ++exact_calls;
            if (!std::isfinite(distance)) {
                throw std::runtime_error("native FCL produced a non-finite distance");
            }
            const bool new_pair = pair != current_pair;
            if (new_pair) {
                current_pair = pair;
                output_pairs.push_back(pair);
                output_first.push_back(first);
                output_second.push_back(second);
                output_distances.push_back(distance);
                for (int axis = 0; axis < 3; ++axis) {
                    output_first_points.push_back(result.nearest_points[0][axis]);
                    output_second_points.push_back(result.nearest_points[1][axis]);
                    output_fallback_normals.push_back(
                        translation(first, axis) - translation(second, axis)
                    );
                }
            } else if (distance < output_distances.back()) {
                output_first.back() = first;
                output_second.back() = second;
                output_distances.back() = distance;
                for (int axis = 0; axis < 3; ++axis) {
                    output_first_points[output_first_points.size() - 3 + axis]
                        = result.nearest_points[0][axis];
                    output_second_points[output_second_points.size() - 3 + axis]
                        = result.nearest_points[1][axis];
                    output_fallback_normals[output_fallback_normals.size() - 3 + axis]
                        = translation(first, axis) - translation(second, axis);
                }
            }
        }
        const py::ssize_t count = output_pairs.size();
        py::array_t<long long> pairs(count), first(count), second(count);
        py::array_t<double> distances(count);
        py::array_t<double> first_points({count, py::ssize_t(3)});
        py::array_t<double> second_points({count, py::ssize_t(3)});
        py::array_t<double> fallback_normals({count, py::ssize_t(3)});
        std::copy(output_pairs.begin(), output_pairs.end(), pairs.mutable_data());
        std::copy(output_first.begin(), output_first.end(), first.mutable_data());
        std::copy(output_second.begin(), output_second.end(), second.mutable_data());
        std::copy(output_distances.begin(), output_distances.end(), distances.mutable_data());
        std::copy(output_first_points.begin(), output_first_points.end(), first_points.mutable_data());
        std::copy(output_second_points.begin(), output_second_points.end(), second_points.mutable_data());
        std::copy(
            output_fallback_normals.begin(), output_fallback_normals.end(),
            fallback_normals.mutable_data()
        );
        return py::make_tuple(
            pairs, first, second, distances, first_points, second_points,
            fallback_normals, exact_calls
        );
    }

 private:
    std::vector<std::shared_ptr<hpp::fcl::CollisionGeometry>> geometries_;
    std::vector<long long> pair_indices_;
    std::vector<long long> first_indices_;
    std::vector<long long> second_indices_;
};

class NativeCollisionKernelV2 {
 public:
    NativeCollisionKernelV2(
        py::array_t<long long, py::array::c_style | py::array::forcecast> kinds,
        py::array_t<double, py::array::c_style | py::array::forcecast> dimensions,
        py::array_t<long long, py::array::c_style | py::array::forcecast> combo_pairs,
        py::array_t<long long, py::array::c_style | py::array::forcecast> combo_first,
        py::array_t<long long, py::array::c_style | py::array::forcecast> combo_second,
        py::array_t<long long, py::array::c_style | py::array::forcecast> pair_first_links,
        py::array_t<long long, py::array::c_style | py::array::forcecast> pair_second_links,
        py::array_t<double, py::array::c_style | py::array::forcecast> local_link_centers,
        py::array_t<double, py::array::c_style | py::array::forcecast> local_link_extents,
        py::array_t<double, py::array::c_style | py::array::forcecast> link_radii
    ) {
        const auto kind = kinds.unchecked<1>();
        const auto size = dimensions.unchecked<2>();
        const auto pairs = combo_pairs.unchecked<1>();
        const auto first = combo_first.unchecked<1>();
        const auto second = combo_second.unchecked<1>();
        const auto first_links = pair_first_links.unchecked<1>();
        const auto second_links = pair_second_links.unchecked<1>();
        const auto centers = local_link_centers.unchecked<2>();
        const auto extents = local_link_extents.unchecked<2>();
        const auto radii = link_radii.unchecked<1>();
        if (
            size.shape(0) != kind.shape(0) || size.shape(1) != 3
            || pairs.shape(0) != first.shape(0) || pairs.shape(0) != second.shape(0)
            || first_links.shape(0) != second_links.shape(0)
            || centers.shape(0) != extents.shape(0) || centers.shape(0) != radii.shape(0)
            || centers.shape(1) != 3 || extents.shape(1) != 3
        ) {
            throw std::invalid_argument("incompatible fused collision kernel inputs");
        }
        pair_count_ = first_links.shape(0);
        link_count_ = centers.shape(0);
        geometry_count_ = kind.shape(0);
        kinds_.reserve(geometry_count_);
        dimensions_.reserve(3 * geometry_count_);
        geometries_.reserve(geometry_count_);
        for (py::ssize_t index = 0; index < geometry_count_; ++index) {
            const auto code = kind(index);
            kinds_.push_back(code);
            for (int axis = 0; axis < 3; ++axis) dimensions_.push_back(size(index, axis));
            if (code == 0) {
                geometries_.push_back(std::make_shared<hpp::fcl::Sphere>(size(index, 0)));
            } else if (code == 1) {
                geometries_.push_back(
                    std::make_shared<hpp::fcl::Capsule>(size(index, 0), size(index, 1))
                );
            } else if (code == 2) {
                geometries_.push_back(std::make_shared<hpp::fcl::Ellipsoid>(
                    size(index, 0), size(index, 1), size(index, 2)
                ));
            } else {
                throw std::invalid_argument("unsupported fused collision geometry kind");
            }
        }
        pair_offsets_.assign(pair_count_ + 1, 0);
        long long previous_pair = -1;
        for (py::ssize_t index = 0; index < pairs.shape(0); ++index) {
            const auto pair = pairs(index);
            if (pair < 0 || pair >= pair_count_ || pair < previous_pair
                || first(index) < 0 || first(index) >= geometry_count_
                || second(index) < 0 || second(index) >= geometry_count_) {
                throw std::invalid_argument("invalid or unordered fused collision combinations");
            }
            previous_pair = pair;
            ++pair_offsets_[pair + 1];
            combo_first_.push_back(first(index));
            combo_second_.push_back(second(index));
        }
        for (py::ssize_t pair = 0; pair < pair_count_; ++pair) {
            pair_offsets_[pair + 1] += pair_offsets_[pair];
            if (pair_offsets_[pair + 1] == pair_offsets_[pair]) {
                throw std::invalid_argument("each logical pair requires primitive combinations");
            }
            pair_first_links_.push_back(first_links(pair));
            pair_second_links_.push_back(second_links(pair));
            if (first_links(pair) < 0 || first_links(pair) >= link_count_
                || second_links(pair) < 0 || second_links(pair) >= link_count_) {
                throw std::invalid_argument("logical pair link index is outside the kernel");
            }
        }
        local_link_centers_.reserve(3 * link_count_);
        local_link_extents_.reserve(3 * link_count_);
        link_radii_.reserve(link_count_);
        for (py::ssize_t link = 0; link < link_count_; ++link) {
            for (int axis = 0; axis < 3; ++axis) {
                local_link_centers_.push_back(centers(link, axis));
                local_link_extents_.push_back(extents(link, axis));
            }
            link_radii_.push_back(radii(link));
        }
    }

    py::tuple compute(
        py::array_t<double, py::array::c_style | py::array::forcecast> thresholds,
        py::array_t<double, py::array::c_style | py::array::forcecast> cached_lower_bounds,
        py::array_t<double, py::array::c_style | py::array::forcecast> link_rotations,
        py::array_t<double, py::array::c_style | py::array::forcecast> link_translations,
        py::array_t<double, py::array::c_style | py::array::forcecast> primitive_rotations,
        py::array_t<double, py::array::c_style | py::array::forcecast> primitive_translations,
        bool prune_combinations = true
    ) const {
        const auto threshold = thresholds.unchecked<1>();
        const auto cached = cached_lower_bounds.unchecked<1>();
        const auto link_rotation = link_rotations.unchecked<3>();
        const auto link_translation = link_translations.unchecked<2>();
        const auto primitive_rotation = primitive_rotations.unchecked<3>();
        const auto primitive_translation = primitive_translations.unchecked<2>();
        if (
            threshold.shape(0) != pair_count_ || cached.shape(0) != pair_count_
            || link_rotation.shape(0) != link_count_ || link_rotation.shape(1) != 3
            || link_rotation.shape(2) != 3 || link_translation.shape(0) != link_count_
            || link_translation.shape(1) != 3
            || primitive_rotation.shape(0) != geometry_count_
            || primitive_rotation.shape(1) != 3 || primitive_rotation.shape(2) != 3
            || primitive_translation.shape(0) != geometry_count_
            || primitive_translation.shape(1) != 3
        ) {
            throw std::invalid_argument("invalid fused collision runtime arrays");
        }
        for (py::ssize_t pair = 0; pair < pair_count_; ++pair) {
            if (!std::isfinite(threshold(pair)) || !std::isfinite(cached(pair))) {
                throw std::invalid_argument("fused collision bounds must be finite");
            }
        }
        const auto broadphase_start = std::chrono::steady_clock::now();
        std::vector<double> link_centers(3 * link_count_);
        std::vector<double> link_extents(3 * link_count_);
        for (py::ssize_t link = 0; link < link_count_; ++link) {
            for (int row = 0; row < 3; ++row) {
                double center = link_translation(link, row);
                double extent = 0.0;
                for (int column = 0; column < 3; ++column) {
                    center += link_rotation(link, row, column)
                        * local_link_centers_[3 * link + column];
                    extent += std::abs(link_rotation(link, row, column))
                        * local_link_extents_[3 * link + column];
                }
                link_centers[3 * link + row] = center;
                link_extents[3 * link + row] = extent;
            }
        }
        std::vector<double> lower_bounds(pair_count_);
        std::vector<unsigned char> aabb_mask(pair_count_, 0);
        long long sphere_count = 0;
        long long aabb_count = 0;
        for (py::ssize_t pair = 0; pair < pair_count_; ++pair) {
            const auto first_link = pair_first_links_[pair];
            const auto second_link = pair_second_links_[pair];
            double squared = 0.0;
            for (int axis = 0; axis < 3; ++axis) {
                const double delta = link_centers[3 * first_link + axis]
                    - link_centers[3 * second_link + axis];
                squared += delta * delta;
            }
            double lower = std::max(
                0.0,
                std::sqrt(squared) - link_radii_[first_link] - link_radii_[second_link]
            );
            lower = std::max(lower, cached(pair));
            const double conservative_threshold = std::nextafter(
                threshold(pair), std::numeric_limits<double>::infinity()
            );
            if (lower <= conservative_threshold) {
                ++sphere_count;
                double separation_squared = 0.0;
                for (int axis = 0; axis < 3; ++axis) {
                    const double first_low = link_centers[3 * first_link + axis]
                        - link_extents[3 * first_link + axis];
                    const double first_high = link_centers[3 * first_link + axis]
                        + link_extents[3 * first_link + axis];
                    const double second_low = link_centers[3 * second_link + axis]
                        - link_extents[3 * second_link + axis];
                    const double second_high = link_centers[3 * second_link + axis]
                        + link_extents[3 * second_link + axis];
                    const double separation = std::max(
                        0.0, std::max(second_low - first_high, first_low - second_high)
                    );
                    separation_squared += separation * separation;
                }
                lower = std::max(lower, std::sqrt(separation_squared));
                if (lower <= conservative_threshold) {
                    aabb_mask[pair] = 1;
                    ++aabb_count;
                }
            }
            lower_bounds[pair] = std::nextafter(
                lower, -std::numeric_limits<double>::infinity()
            );
        }

        std::vector<double> primitive_extents(3 * geometry_count_);
        for (py::ssize_t geometry = 0; geometry < geometry_count_; ++geometry) {
            const auto kind = kinds_[geometry];
            for (int row = 0; row < 3; ++row) {
                double extent = 0.0;
                if (kind == 0) {
                    extent = dimensions_[3 * geometry];
                } else if (kind == 1) {
                    extent = dimensions_[3 * geometry]
                        + std::abs(primitive_rotation(geometry, row, 2))
                        * 0.5 * dimensions_[3 * geometry + 1];
                } else {
                    for (int column = 0; column < 3; ++column) {
                        extent += std::abs(primitive_rotation(geometry, row, column))
                            * dimensions_[3 * geometry + column];
                    }
                }
                primitive_extents[3 * geometry + row] = extent;
            }
        }
        std::vector<double> combination_lower(combo_first_.size(), 0.0);
        std::vector<unsigned char> active_mask(pair_count_, 0);
        for (py::ssize_t pair = 0; pair < pair_count_; ++pair) {
            if (!aabb_mask[pair]) continue;
            double primitive_lower = std::numeric_limits<double>::infinity();
            for (long long combo = pair_offsets_[pair]; combo < pair_offsets_[pair + 1]; ++combo) {
                const auto first = combo_first_[combo];
                const auto second = combo_second_[combo];
                double separation_squared = 0.0;
                for (int axis = 0; axis < 3; ++axis) {
                    const double first_low = primitive_translation(first, axis)
                        - primitive_extents[3 * first + axis];
                    const double first_high = primitive_translation(first, axis)
                        + primitive_extents[3 * first + axis];
                    const double second_low = primitive_translation(second, axis)
                        - primitive_extents[3 * second + axis];
                    const double second_high = primitive_translation(second, axis)
                        + primitive_extents[3 * second + axis];
                    const double separation = std::max(
                        0.0, std::max(second_low - first_high, first_low - second_high)
                    );
                    separation_squared += separation * separation;
                }
                const double bound = std::nextafter(std::sqrt(separation_squared),
                    -std::numeric_limits<double>::infinity());
                combination_lower[combo] = bound;
                primitive_lower = std::min(primitive_lower, bound);
            }
            lower_bounds[pair] = std::nextafter(
                std::max(lower_bounds[pair], primitive_lower),
                -std::numeric_limits<double>::infinity()
            );
            if (lower_bounds[pair] <= std::nextafter(
                    threshold(pair), std::numeric_limits<double>::infinity())) {
                active_mask[pair] = 1;
            }
        }
        const auto broadphase_end = std::chrono::steady_clock::now();

        std::vector<hpp::fcl::Transform3f> transforms(geometry_count_);
        for (py::ssize_t geometry = 0; geometry < geometry_count_; ++geometry) {
            hpp::fcl::Matrix3f matrix;
            hpp::fcl::Vec3f vector;
            for (int row = 0; row < 3; ++row) {
                vector[row] = primitive_translation(geometry, row);
                for (int column = 0; column < 3; ++column) {
                    matrix(row, column) = primitive_rotation(geometry, row, column);
                }
            }
            transforms[geometry].setTransform(matrix, vector);
        }
        const auto exact_start = std::chrono::steady_clock::now();
        std::vector<long long> result_pairs;
        std::vector<long long> result_first;
        std::vector<long long> result_second;
        std::vector<double> result_distances;
        std::vector<double> result_first_points;
        std::vector<double> result_second_points;
        std::vector<double> result_fallback_normals;
        std::vector<unsigned char> result_barrier;
        result_pairs.reserve(pair_count_);
        result_first.reserve(pair_count_);
        result_second.reserve(pair_count_);
        result_distances.reserve(pair_count_);
        result_first_points.reserve(3 * pair_count_);
        result_second_points.reserve(3 * pair_count_);
        result_fallback_normals.reserve(3 * pair_count_);
        result_barrier.reserve(pair_count_);
        long long analytic_calls = 0;
        long long fcl_calls = 0;
        hpp::fcl::DistanceRequest request(true);
        for (py::ssize_t pair = 0; pair < pair_count_; ++pair) {
            if (!active_mask[pair]) continue;
            double best_distance = std::numeric_limits<double>::infinity();
            long long best_first = -1;
            long long best_second = -1;
            double best_first_point[3] = {};
            double best_second_point[3] = {};
            double best_fallback[3] = {};
            for (long long combo = pair_offsets_[pair]; combo < pair_offsets_[pair + 1]; ++combo) {
                const auto first = combo_first_[combo];
                const auto second = combo_second_[combo];
                // A strictly separated primitive cannot improve the current
                // exact union witness. Overlapping AABBs always retain exact
                // evaluation, including negative penetration distances.
                if (prune_combinations && combination_lower[combo] > 0.0
                    && combination_lower[combo] > best_distance + 1e-9) continue;
                double distance;
                double first_point[3];
                double second_point[3];
                double fallback[3];
                if (kinds_[first] != 2 && kinds_[second] != 2) {
                    double first_start[3];
                    double first_end[3];
                    double second_start[3];
                    double second_end[3];
                    for (int axis = 0; axis < 3; ++axis) {
                        const double first_half = kinds_[first] == 1
                            ? 0.5 * dimensions_[3 * first + 1]
                                * primitive_rotation(first, axis, 2)
                            : 0.0;
                        const double second_half = kinds_[second] == 1
                            ? 0.5 * dimensions_[3 * second + 1]
                                * primitive_rotation(second, axis, 2)
                            : 0.0;
                        first_start[axis] = primitive_translation(first, axis) - first_half;
                        first_end[axis] = primitive_translation(first, axis) + first_half;
                        second_start[axis] = primitive_translation(second, axis) - second_half;
                        second_end[axis] = primitive_translation(second, axis) + second_half;
                    }
                    double first_axis_point[3];
                    double second_axis_point[3];
                    closest_segment_points(
                        first_start, first_end, second_start, second_end,
                        first_axis_point, second_axis_point
                    );
                    double squared = 0.0;
                    for (int axis = 0; axis < 3; ++axis) {
                        fallback[axis] = first_axis_point[axis] - second_axis_point[axis];
                        squared += fallback[axis] * fallback[axis];
                    }
                    double norm = std::sqrt(squared);
                    if (norm <= 1e-12) {
                        squared = 0.0;
                        for (int axis = 0; axis < 3; ++axis) {
                            fallback[axis] = primitive_translation(first, axis)
                                - primitive_translation(second, axis);
                            squared += fallback[axis] * fallback[axis];
                        }
                        norm = std::sqrt(squared);
                    }
                    distance = norm - dimensions_[3 * first] - dimensions_[3 * second];
                    for (int axis = 0; axis < 3; ++axis) {
                        const double normal = norm > 1e-12
                            ? fallback[axis] / norm : (axis == 0 ? 1.0 : 0.0);
                        first_point[axis] = first_axis_point[axis]
                            - normal * dimensions_[3 * first];
                        second_point[axis] = second_axis_point[axis]
                            + normal * dimensions_[3 * second];
                    }
                    ++analytic_calls;
                } else {
                    hpp::fcl::DistanceResult result;
                    distance = hpp::fcl::distance(
                        geometries_[first].get(), transforms[first],
                        geometries_[second].get(), transforms[second], request, result
                    );
                    for (int axis = 0; axis < 3; ++axis) {
                        first_point[axis] = result.nearest_points[0][axis];
                        second_point[axis] = result.nearest_points[1][axis];
                        fallback[axis] = primitive_translation(first, axis)
                            - primitive_translation(second, axis);
                    }
                    ++fcl_calls;
                }
                if (!std::isfinite(distance)) {
                    throw std::runtime_error("fused collision kernel produced a non-finite distance");
                }
                if (distance < best_distance) {
                    best_distance = distance;
                    best_first = first;
                    best_second = second;
                    std::copy(first_point, first_point + 3, best_first_point);
                    std::copy(second_point, second_point + 3, best_second_point);
                    std::copy(fallback, fallback + 3, best_fallback);
                }
            }
            result_pairs.push_back(pair);
            result_first.push_back(best_first);
            result_second.push_back(best_second);
            result_distances.push_back(best_distance);
            result_first_points.insert(
                result_first_points.end(), best_first_point, best_first_point + 3
            );
            result_second_points.insert(
                result_second_points.end(), best_second_point, best_second_point + 3
            );
            result_fallback_normals.insert(
                result_fallback_normals.end(), best_fallback, best_fallback + 3
            );
            result_barrier.push_back(best_distance <= std::nextafter(
                threshold(pair), std::numeric_limits<double>::infinity()
            ));
            lower_bounds[pair] = std::nextafter(
                std::max(lower_bounds[pair], best_distance),
                -std::numeric_limits<double>::infinity()
            );
        }
        const auto exact_end = std::chrono::steady_clock::now();
        const py::ssize_t count = result_pairs.size();
        py::array_t<long long> pairs(count), first(count), second(count);
        py::array_t<double> distances(count), first_points({count, py::ssize_t(3)});
        py::array_t<double> second_points({count, py::ssize_t(3)});
        py::array_t<double> fallback_normals({count, py::ssize_t(3)});
        py::array_t<bool> barrier(count);
        py::array_t<double> lower(pair_count_);
        std::copy(result_pairs.begin(), result_pairs.end(), pairs.mutable_data());
        std::copy(result_first.begin(), result_first.end(), first.mutable_data());
        std::copy(result_second.begin(), result_second.end(), second.mutable_data());
        std::copy(result_distances.begin(), result_distances.end(), distances.mutable_data());
        std::copy(result_first_points.begin(), result_first_points.end(), first_points.mutable_data());
        std::copy(result_second_points.begin(), result_second_points.end(), second_points.mutable_data());
        std::copy(
            result_fallback_normals.begin(), result_fallback_normals.end(),
            fallback_normals.mutable_data()
        );
        std::copy(result_barrier.begin(), result_barrier.end(), barrier.mutable_data());
        std::copy(lower_bounds.begin(), lower_bounds.end(), lower.mutable_data());
        const double broadphase_ms = std::chrono::duration<double, std::milli>(
            broadphase_end - broadphase_start
        ).count();
        const double exact_ms = std::chrono::duration<double, std::milli>(
            exact_end - exact_start
        ).count();
        return py::make_tuple(
            pairs, first, second, distances, first_points, second_points,
            fallback_normals, barrier, lower,
            sphere_count, aabb_count, analytic_calls + fcl_calls,
            analytic_calls, fcl_calls, broadphase_ms, exact_ms
        );
    }

 private:
    py::ssize_t pair_count_ = 0;
    py::ssize_t link_count_ = 0;
    py::ssize_t geometry_count_ = 0;
    std::vector<long long> kinds_;
    std::vector<double> dimensions_;
    std::vector<std::shared_ptr<hpp::fcl::CollisionGeometry>> geometries_;
    std::vector<long long> combo_first_;
    std::vector<long long> combo_second_;
    std::vector<long long> pair_offsets_;
    std::vector<long long> pair_first_links_;
    std::vector<long long> pair_second_links_;
    std::vector<double> local_link_centers_;
    std::vector<double> local_link_extents_;
    std::vector<double> link_radii_;
};

py::array_t<double> witness_jacobians(
    py::array_t<long long, py::array::c_style | py::array::forcecast> first_joint_rows,
    py::array_t<long long, py::array::c_style | py::array::forcecast> second_joint_rows,
    py::array_t<double, py::array::c_style | py::array::forcecast> first_points,
    py::array_t<double, py::array::c_style | py::array::forcecast> second_points,
    py::array_t<double, py::array::c_style | py::array::forcecast> fallback_normals,
    py::array_t<double, py::array::c_style | py::array::forcecast> joint_jacobians,
    py::array_t<double, py::array::c_style | py::array::forcecast> joint_origins
) {
    const auto first_rows = first_joint_rows.unchecked<1>();
    const auto second_rows = second_joint_rows.unchecked<1>();
    const auto first = first_points.unchecked<2>();
    const auto second = second_points.unchecked<2>();
    const auto fallback = fallback_normals.unchecked<2>();
    const auto jacobians = joint_jacobians.unchecked<3>();
    const auto origins = joint_origins.unchecked<2>();
    const auto row_count = first_rows.shape(0);
    const auto joint_count = jacobians.shape(0);
    const auto nv = jacobians.shape(2);
    if (
        second_rows.shape(0) != row_count || first.shape(0) != row_count
        || second.shape(0) != row_count || fallback.shape(0) != row_count
        || first.shape(1) != 3 || second.shape(1) != 3 || fallback.shape(1) != 3
        || jacobians.shape(1) != 6 || origins.shape(0) != joint_count
        || origins.shape(1) != 3
    ) {
        throw std::invalid_argument("invalid witness Jacobian arrays");
    }
    py::array_t<double> output({row_count, nv});
    auto rows = output.mutable_unchecked<2>();
    for (py::ssize_t row = 0; row < row_count; ++row) {
        const auto first_joint = first_rows(row);
        const auto second_joint = second_rows(row);
        if (first_joint < 0 || first_joint >= joint_count
            || second_joint < 0 || second_joint >= joint_count) {
            throw std::invalid_argument("witness parent joint is outside the Jacobian stack");
        }
        double delta[3];
        double squared = 0.0;
        for (int axis = 0; axis < 3; ++axis) {
            delta[axis] = first(row, axis) - second(row, axis);
            squared += delta[axis] * delta[axis];
        }
        if (squared <= 1e-24) {
            squared = 0.0;
            for (int axis = 0; axis < 3; ++axis) {
                delta[axis] = fallback(row, axis);
                squared += delta[axis] * delta[axis];
            }
        }
        const double norm = std::sqrt(squared);
        double normal[3] = {};
        double first_angular[3] = {};
        double second_angular[3] = {};
        if (norm > 1e-12) {
            for (int axis = 0; axis < 3; ++axis) normal[axis] = delta[axis] / norm;
        }
        double first_offset[3];
        double second_offset[3];
        for (int axis = 0; axis < 3; ++axis) {
            first_offset[axis] = first(row, axis) - origins(first_joint, axis);
            second_offset[axis] = second(row, axis) - origins(second_joint, axis);
        }
        first_angular[0] = first_offset[1] * normal[2] - first_offset[2] * normal[1];
        first_angular[1] = first_offset[2] * normal[0] - first_offset[0] * normal[2];
        first_angular[2] = first_offset[0] * normal[1] - first_offset[1] * normal[0];
        second_angular[0] = second_offset[1] * normal[2] - second_offset[2] * normal[1];
        second_angular[1] = second_offset[2] * normal[0] - second_offset[0] * normal[2];
        second_angular[2] = second_offset[0] * normal[1] - second_offset[1] * normal[0];
        for (py::ssize_t column = 0; column < nv; ++column) {
            double value = 0.0;
            for (int axis = 0; axis < 3; ++axis) {
                value += normal[axis] * (
                    jacobians(first_joint, axis, column)
                    - jacobians(second_joint, axis, column)
                );
                value += first_angular[axis] * jacobians(first_joint, axis + 3, column)
                    - second_angular[axis] * jacobians(second_joint, axis + 3, column);
            }
            rows(row, column) = std::isfinite(value) ? value : 0.0;
        }
    }
    return output;
}

void right_jacobian_so3(const double* vector, double matrix[3][3]) {
    const double theta2 = dot3(vector, vector);
    double a;
    double b;
    if (theta2 < 1e-12) {
        // Stable Taylor series for (1-cos(theta))/theta^2 and
        // (theta-sin(theta))/theta^3.
        a = 0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0;
        b = 1.0 / 6.0 - theta2 / 120.0 + theta2 * theta2 / 5040.0;
    } else {
        const double theta = std::sqrt(theta2);
        a = (1.0 - std::cos(theta)) / theta2;
        b = (theta - std::sin(theta)) / (theta2 * theta);
    }
    const double skew[3][3] = {
        {0.0, -vector[2], vector[1]},
        {vector[2], 0.0, -vector[0]},
        {-vector[1], vector[0], 0.0},
    };
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            double skew2 = 0.0;
            for (int inner = 0; inner < 3; ++inner) {
                skew2 += skew[row][inner] * skew[inner][column];
            }
            matrix[row][column] = (row == column ? 1.0 : 0.0)
                - a * skew[row][column] + b * skew2;
        }
    }
}

void rotation_motion(
    const double* vector,
    const double* velocity,
    const double* acceleration,
    double* omega,
    double* alpha
) {
    double jacobian[3][3];
    right_jacobian_so3(vector, jacobian);
    double direction_norm = std::sqrt(dot3(velocity, velocity));
    const double epsilon = 1e-6 / std::max(1.0, direction_norm);
    double plus[3];
    double minus[3];
    for (int axis = 0; axis < 3; ++axis) {
        plus[axis] = vector[axis] + epsilon * velocity[axis];
        minus[axis] = vector[axis] - epsilon * velocity[axis];
    }
    double jacobian_plus[3][3];
    double jacobian_minus[3][3];
    right_jacobian_so3(plus, jacobian_plus);
    right_jacobian_so3(minus, jacobian_minus);
    for (int row = 0; row < 3; ++row) {
        omega[row] = 0.0;
        alpha[row] = 0.0;
        for (int column = 0; column < 3; ++column) {
            omega[row] += jacobian[row][column] * velocity[column];
            alpha[row] += jacobian[row][column] * acceleration[column]
                + (jacobian_plus[row][column] - jacobian_minus[row][column])
                    / (2.0 * epsilon) * velocity[column];
        }
    }
}

std::array<double, 6> quintic_peak_norm_values(
    py::array_t<double, py::array::c_style | py::array::forcecast> translation,
    py::array_t<double, py::array::c_style | py::array::forcecast> rotation,
    double duration,
    int sample_count
) {
    const auto linear = translation.unchecked<2>();
    const auto angular = rotation.unchecked<2>();
    if (
        (linear.shape(0) != 6 && linear.shape(0) != 8) || linear.shape(1) != 3
        || angular.shape(0) != linear.shape(0) || angular.shape(1) != 3
        || !std::isfinite(duration) || duration <= 0.0
        || sample_count < 3
    ) {
        throw std::invalid_argument("invalid quintic limit-check inputs");
    }
    double maximum_rotation_value = 0.0;
    for (int row = 0; row < angular.shape(0); ++row) {
        for (int axis = 0; axis < 3; ++axis) {
            const double value = angular(row, axis);
            if (!std::isfinite(value) || !std::isfinite(linear(row, axis))) {
                throw std::invalid_argument("nonfinite polynomial coefficients");
            }
            maximum_rotation_value = std::max(maximum_rotation_value, std::abs(value));
        }
    }

    const bool zero_rotation = maximum_rotation_value <= 1e-14;
    std::vector<std::array<double, 3>> angular_acceleration(sample_count);
    std::array<double, 6> peaks{};
    for (int sample = 0; sample < sample_count; ++sample) {
        const double t = duration * static_cast<double>(sample)
            / static_cast<double>(sample_count - 1);
        const double t2 = t * t;
        const double t3 = t2 * t;
        const double t4 = t3 * t;
        double norms_squared[5] = {};
        double rotation_vector[3] = {};
        double rotation_velocity[3] = {};
        double rotation_acceleration[3] = {};
        for (int axis = 0; axis < 3; ++axis) {
            double lv = linear(1, axis) + 2.0 * t * linear(2, axis)
                + 3.0 * t2 * linear(3, axis) + 4.0 * t3 * linear(4, axis)
                + 5.0 * t4 * linear(5, axis);
            double la = 2.0 * linear(2, axis) + 6.0 * t * linear(3, axis)
                + 12.0 * t2 * linear(4, axis) + 20.0 * t3 * linear(5, axis);
            double lj = 6.0 * linear(3, axis) + 24.0 * t * linear(4, axis)
                + 60.0 * t2 * linear(5, axis);
            if (linear.shape(0) == 8) {
                lv += 6*t4*t*linear(6,axis) + 7*t4*t2*linear(7,axis);
                la += 30*t4*linear(6,axis) + 42*t4*t*linear(7,axis);
                lj += 120*t3*linear(6,axis) + 210*t4*linear(7,axis);
            }
            norms_squared[0] += lv * lv;
            norms_squared[1] += la * la;
            norms_squared[2] += lj * lj;
            rotation_vector[axis] = angular(0, axis) + t * angular(1, axis)
                + t2 * angular(2, axis) + t3 * angular(3, axis)
                + t4 * angular(4, axis) + t4 * t * angular(5, axis);
            rotation_velocity[axis] = angular(1, axis) + 2.0 * t * angular(2, axis)
                + 3.0 * t2 * angular(3, axis) + 4.0 * t3 * angular(4, axis)
                + 5.0 * t4 * angular(5, axis);
            rotation_acceleration[axis] = 2.0 * angular(2, axis)
                + 6.0 * t * angular(3, axis) + 12.0 * t2 * angular(4, axis)
                + 20.0 * t3 * angular(5, axis);
            if (angular.shape(0) == 8) {
                rotation_vector[axis] += t4*t2*angular(6,axis) + t4*t3*angular(7,axis);
                rotation_velocity[axis] += 6*t4*t*angular(6,axis) + 7*t4*t2*angular(7,axis);
                rotation_acceleration[axis] += 30*t4*angular(6,axis) + 42*t4*t*angular(7,axis);
            }
        }
        double omega[3] = {};
        double alpha[3] = {};
        if (!zero_rotation) {
            rotation_motion(rotation_vector, rotation_velocity, rotation_acceleration, omega, alpha);
        }
        for (int axis = 0; axis < 3; ++axis) {
            norms_squared[3] += omega[axis] * omega[axis];
            norms_squared[4] += alpha[axis] * alpha[axis];
            angular_acceleration[sample][axis] = alpha[axis];
        }
        for (int kind = 0; kind < 5; ++kind) {
            peaks[kind] = std::max(peaks[kind], std::sqrt(norms_squared[kind]));
        }
    }
    const double step = duration / static_cast<double>(sample_count - 1);
    for (int sample = 0; sample < sample_count; ++sample) {
        double jerk_squared = 0.0;
        for (int axis = 0; axis < 3; ++axis) {
            double jerk;
            if (sample == 0) {
                jerk = (-3.0 * angular_acceleration[0][axis]
                    + 4.0 * angular_acceleration[1][axis]
                    - angular_acceleration[2][axis]) / (2.0 * step);
            } else if (sample == sample_count - 1) {
                jerk = (3.0 * angular_acceleration[sample][axis]
                    - 4.0 * angular_acceleration[sample - 1][axis]
                    + angular_acceleration[sample - 2][axis]) / (2.0 * step);
            } else {
                jerk = (angular_acceleration[sample + 1][axis]
                    - angular_acceleration[sample - 1][axis]) / (2.0 * step);
            }
            jerk_squared += jerk * jerk;
        }
        peaks[5] = std::max(peaks[5], std::sqrt(jerk_squared));
    }
    // Round peaks toward the unsafe side so native arithmetic cannot turn a
    // boundary violation into an accepted trajectory.
    for (double& peak : peaks) {
        peak = std::nextafter(peak, std::numeric_limits<double>::infinity());
    }
    return peaks;
}

py::array_t<double> quintic_peak_norms(
    py::array_t<double, py::array::c_style | py::array::forcecast> translation,
    py::array_t<double, py::array::c_style | py::array::forcecast> rotation,
    double duration,
    int sample_count
) {
    const auto peaks = quintic_peak_norm_values(translation, rotation, duration, sample_count);
    py::array_t<double> output(6);
    auto values = output.mutable_unchecked<1>();
    for (int index = 0; index < 6; ++index) values(index) = peaks[index];
    return output;
}

py::bool_ quintic_limits_are_satisfied(
    py::array_t<double, py::array::c_style | py::array::forcecast> translation,
    py::array_t<double, py::array::c_style | py::array::forcecast> rotation,
    double duration,
    py::array_t<double, py::array::c_style | py::array::forcecast> limits,
    int sample_count
) {
    const auto maximum = limits.unchecked<1>();
    if (maximum.shape(0) != 6) throw std::invalid_argument("invalid quintic limits");
    const auto peaks = quintic_peak_norm_values(translation, rotation, duration, sample_count);
    constexpr double tolerance_scale = 1.0 + 1e-6;
    for (int index = 0; index < 6; ++index) {
        if (!std::isfinite(maximum(index)) || maximum(index) <= 0.0) {
            throw std::invalid_argument("quintic limits must be finite and positive");
        }
        if (peaks[index] > maximum(index) * tolerance_scale) return py::bool_(false);
    }
    return py::bool_(true);
}

// Additive trajectory API; collision ABI and its conservative checks are unchanged.
py::array_t<double> quintic_coefficients(
    py::array_t<double, py::array::c_style | py::array::forcecast> boundaries,
    double duration
) {
    auto b = boundaries.unchecked<2>();
    if (b.shape(0) != 6 || b.shape(1) != 6 || !std::isfinite(duration) || duration <= 0.)
        throw std::invalid_argument("quintic boundaries require shape (6,6) and positive duration");
    py::array_t<double> result({6, 6});
    auto c = result.mutable_unchecked<2>();
    const double t = duration, t2 = t*t, t3 = t2*t, t4 = t3*t, t5 = t4*t;
    for (int i=0; i<6; ++i) {
        for (int row=0; row<6; ++row)
            if (!std::isfinite(b(row,i))) throw std::invalid_argument("nonfinite quintic boundary");
        c(0,i)=b(0,i); c(1,i)=b(1,i); c(2,i)=.5*b(2,i);
        const double p=b(3,i)-(c(0,i)+c(1,i)*t+c(2,i)*t2);
        const double v=b(4,i)-(c(1,i)+2*c(2,i)*t);
        const double a=b(5,i)-2*c(2,i);
        c(3,i)=(10*p-4*t*v+.5*t2*a)/t3;
        c(4,i)=(-15*p+7*t*v-t2*a)/t4;
        c(5,i)=(6*p-3*t*v+.5*t2*a)/t5;
    }
    return result;
}

py::array_t<double> reference_future_bounds(
    py::array_t<double, py::array::c_style | py::array::forcecast> distances,
    py::array_t<double, py::array::c_style | py::array::forcecast> velocities,
    py::array_t<double, py::array::c_style | py::array::forcecast> maximum,
    py::array_t<double, py::array::c_style | py::array::forcecast> acceleration,
    py::array_t<double, py::array::c_style | py::array::forcecast> jerk,
    double dt, int horizon
) {
    auto d=distances.unchecked<1>(), v=velocities.unchecked<1>(), m=maximum.unchecked<1>();
    auto a=acceleration.unchecked<1>(), j=jerk.unchecked<1>();
    const int n=d.shape(0), half=a.shape(0);
    if (n!=2*half || v.shape(0)!=n || m.shape(0)!=n || j.shape(0)!=half
        || !std::isfinite(dt) || dt<=0 || horizon<1)
        throw std::invalid_argument("invalid reference bound dimensions");
    py::array_t<double> result(n); auto out=result.mutable_unchecked<1>();
    for(int axis=0; axis<n; ++axis) {
        const double am=a(axis%half), jm=j(axis%half), previous=v(axis);
        if (!std::isfinite(d(axis)) || !std::isfinite(previous) || !std::isfinite(m(axis))
            || m(axis)<0 || !std::isfinite(am) || am<=0 || !std::isfinite(jm) || jm<=0)
            throw std::invalid_argument("invalid reference bounds");
        const double distance=std::max(d(axis)-.002,0.);
        const double ramp=2*am/jm, constant=am*dt*dt+am*dt*ramp+2*am*am*am/(3*jm*jm);
        const double linear=2*dt+ramp;
        auto fits = [&](double candidate) {
            double integrated=0., travel=0.;
            for(int k=0; k<horizon; ++k) {
                if(k) integrated+=std::max((candidate-previous)/dt-k*jm*dt,-am);
                const double speed=std::max(candidate+dt*integrated,0.);
                const double envelope=speed>1e-12 ? speed*speed/(2*am)+linear*speed+constant : 0.;
                if(travel+envelope>distance) return false;
                travel+=dt*speed;
            }
            return true;
        };
        if(fits(m(axis))) { out(axis)=m(axis); continue; }
        double low=0., high=m(axis);
        for(int it=0; it<36 && high-low>1e-12; ++it) {
            const double middle=.5*(low+high);
            if(fits(middle)) low=middle; else high=middle;
        }
        out(axis)=low;
    }
    return result;
}

}  // namespace

PYBIND11_MODULE(_hexfellow_collision_kernel, module) {
    module.doc() = "Conservative capsule/sphere union distance kernel";
    module.def("segment_union_distances", &segment_union_distances);
    module.def("quintic_limits_are_satisfied", &quintic_limits_are_satisfied);
    module.def("quintic_peak_norms", &quintic_peak_norms);
    module.def("quintic_coefficients", &quintic_coefficients);
    module.def("reference_future_bounds", &reference_future_bounds);
    module.attr("TRAJECTORY_KERNEL_VERSION") = 1;
    module.attr("KERNEL_ABI_VERSION") = kKernelAbiVersion;
    module.def("witness_jacobians", &witness_jacobians);
    py::class_<NativeFclUnionKernel>(module, "NativeFclUnionKernel")
        .def(py::init<
            py::array_t<long long, py::array::c_style | py::array::forcecast>,
            py::array_t<double, py::array::c_style | py::array::forcecast>,
            py::array_t<long long, py::array::c_style | py::array::forcecast>,
            py::array_t<long long, py::array::c_style | py::array::forcecast>,
            py::array_t<long long, py::array::c_style | py::array::forcecast>
        >())
        .def("compute", &NativeFclUnionKernel::compute);
    py::class_<NativeCollisionKernelV2>(module, "NativeCollisionKernelV2")
        .def(py::init<
            py::array_t<long long, py::array::c_style | py::array::forcecast>,
            py::array_t<double, py::array::c_style | py::array::forcecast>,
            py::array_t<long long, py::array::c_style | py::array::forcecast>,
            py::array_t<long long, py::array::c_style | py::array::forcecast>,
            py::array_t<long long, py::array::c_style | py::array::forcecast>,
            py::array_t<long long, py::array::c_style | py::array::forcecast>,
            py::array_t<long long, py::array::c_style | py::array::forcecast>,
            py::array_t<double, py::array::c_style | py::array::forcecast>,
            py::array_t<double, py::array::c_style | py::array::forcecast>,
            py::array_t<double, py::array::c_style | py::array::forcecast>
        >())
        .def("compute", &NativeCollisionKernelV2::compute,
             py::arg("thresholds"), py::arg("cached_lower_bounds"),
             py::arg("link_rotations"), py::arg("link_translations"),
             py::arg("primitive_rotations"), py::arg("primitive_translations"),
             py::arg("prune_combinations") = true);
}
