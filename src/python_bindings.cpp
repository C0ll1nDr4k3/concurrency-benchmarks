#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "common.hpp"
#include "flat_vanilla.hpp"
#include "hnsw_coarse_optimistic.hpp"
#include "hnsw_coarse_pessimistic.hpp"
#include "hnsw_fine_optimistic.hpp"
#include "hnsw_fine_pessimistic.hpp"
#include "hnsw_vanilla.hpp"
#include "ivfflat_coarse_optimistic.hpp"
#include "ivfflat_coarse_pessimistic.hpp"
#include "ivfflat_fine_optimistic.hpp"
#include "ivfflat_fine_pessimistic.hpp"
#include "ivfflat_vanilla.hpp"

namespace py = pybind11;
using namespace nilvec;

PYBIND11_MODULE(_nilvec, m) {
  m.doc() = "Python bindings for NilVec ANN indexes";

  // Bind SearchResult
  py::class_<SearchResult>(m, "SearchResult")
      .def_readonly("ids", &SearchResult::ids)
      .def_readonly("distances", &SearchResult::distances)
      .def("__repr__", [](const SearchResult& self) {
        return "<SearchResult ids=" + std::to_string(self.ids.size()) + ">";
      });

  // Bind ConflictStats
  py::class_<ConflictStats>(m, "ConflictStats")
      .def("insert_conflict_rate", &ConflictStats::insert_conflict_rate)
      .def("search_conflict_rate", &ConflictStats::search_conflict_rate)
      .def("reset", &ConflictStats::reset)
      .def_property_readonly(
          "insert_attempts",
          [](const ConflictStats& s) { return s.insert_attempts.load(); })
      .def_property_readonly(
          "insert_conflicts",
          [](const ConflictStats& s) { return s.insert_conflicts.load(); })
      .def_property_readonly(
          "search_attempts",
          [](const ConflictStats& s) { return s.search_attempts.load(); })
      .def_property_readonly("search_conflicts", [](const ConflictStats& s) {
        return s.search_conflicts.load();
      });

  // --- Flat Index ---
  py::class_<FlatVanilla<float>>(m, "FlatVanilla")
      .def(py::init<Dim>(), py::arg("dim"))
      .def("insert",
           [](FlatVanilla<float>& self, const std::vector<float>& data) {
             return self.insert(std::span<const float>(data));
           })
      .def("search",
           [](const FlatVanilla<float>& self, const std::vector<float>& query,
              size_t k) {
             return self.search(std::span<const float>(query), k);
           })
      .def("size", &FlatVanilla<float>::size)
      .def("dim", &FlatVanilla<float>::dim);

  // --- HNSW Indexes ---
  // Helper to bind HNSW classes
  auto bind_hnsw = [&](auto tag, const std::string& name,
                       bool has_stats = false) {
    using Index = typename decltype(tag)::type;
    auto cls =
        py::class_<Index>(m, name.c_str())
            .def(py::init<Dim, size_t, size_t, float>(), py::arg("dim"),
                 py::arg("M") = 16, py::arg("ef_construction") = 200,
                 py::arg("mL") = 0.0f)
            .def(
                "insert",
                [](Index& self, const std::vector<float>& data) {
                  return self.insert(std::span<const float>(data));
                },
                py::call_guard<py::gil_scoped_release>())
            .def(
                "search",
                [](const Index& self, const std::vector<float>& query, size_t k,
                   size_t ef) {
                  return self.search(std::span<const float>(query), k, ef);
                },
                py::arg("query"), py::arg("k"), py::arg("ef") = 0,
                py::call_guard<py::gil_scoped_release>())
            .def("size", &Index::size)
            .def("max_level", &Index::max_level);

    if (has_stats) {
      // We need to use a lambda or similar if the method might not exist in
      // template instantiation, but here we know these specific classes have it
      // or we wouldn't set has_stats. However, to be safe with templates in
      // C++, we use if constexpr in a lambda.
      cls.def(
          "conflict_stats",
          [](const Index& self) -> const ConflictStats& {
            if constexpr (requires { self.conflict_stats(); }) {
              return self.conflict_stats();
            } else {
              static ConflictStats empty;
              return empty;
            }
          },
          py::return_value_policy::reference);
    }
  };

  bind_hnsw(std::type_identity<HNSWVanilla<float>>{}, "HNSWVanilla", false);
  bind_hnsw(std::type_identity<HNSWCoarseOptimistic<float>>{},
            "HNSWCoarseOptimistic", true);
  bind_hnsw(std::type_identity<HNSWCoarsePessimistic<float>>{},
            "HNSWCoarsePessimistic", true);
  bind_hnsw(std::type_identity<HNSWFineOptimistic<float>>{},
            "HNSWFineOptimistic", true);
  bind_hnsw(std::type_identity<HNSWFinePessimistic<float>>{},
            "HNSWFinePessimistic", true);

  // --- IVFFlat Indexes ---
  // Helper to bind IVFFlat classes
  auto bind_ivfflat = [&](auto tag, const std::string& name,
                          bool has_stats = false) {
    using Index = typename decltype(tag)::type;
    auto cls =
        py::class_<Index>(m, name.c_str())
            .def(py::init<Dim, size_t, size_t>(), py::arg("dim"),
                 py::arg("nlist") = 100, py::arg("nprobe") = 1)
            .def(
                "train",
                [](Index& self, const std::vector<std::vector<float>>& data) {
                  self.train(data);
                },
                py::call_guard<py::gil_scoped_release>())
            .def(
                "insert",
                [](Index& self, const std::vector<float>& data) {
                  return self.insert(std::span<const float>(data));
                },
                py::call_guard<py::gil_scoped_release>())
            .def(
                "search",
                [](const Index& self, const std::vector<float>& query,
                   size_t k) {
                  return self.search(std::span<const float>(query), k);
                },
                py::call_guard<py::gil_scoped_release>())
            .def("size", &Index::size)
            .def("is_trained", &Index::is_trained)
            .def("set_nprobe", &Index::set_nprobe)
            .def("nlist", &Index::nlist);

    if (has_stats) {
      cls.def(
          "conflict_stats",
          [](const Index& self) -> const ConflictStats& {
            if constexpr (requires { self.conflict_stats(); }) {
              return self.conflict_stats();
            } else {
              static ConflictStats empty;
              return empty;
            }
          },
          py::return_value_policy::reference);
    }
  };

  bind_ivfflat(std::type_identity<IVFFlatVanilla<float>>{}, "IVFFlatVanilla",
               false);
  bind_ivfflat(std::type_identity<IVFFlatCoarseOptimistic<float>>{},
               "IVFFlatCoarseOptimistic", true);
  bind_ivfflat(std::type_identity<IVFFlatCoarsePessimistic<float>>{},
               "IVFFlatCoarsePessimistic", true);
  bind_ivfflat(std::type_identity<IVFFlatFineOptimistic<float>>{},
               "IVFFlatFineOptimistic", true);
  bind_ivfflat(std::type_identity<IVFFlatFinePessimistic<float>>{},
               "IVFFlatFinePessimistic", true);
}
