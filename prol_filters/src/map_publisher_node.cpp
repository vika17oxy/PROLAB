/**
 * map_publisher_node.cpp — publishes PROL_Vika's map.yaml/.pgm as a latched
 * nav_msgs/OccupancyGrid on /map (the map backdrop). Minimal YAML
 * + PGM (P5/P2) parsing, no external deps.
 */
#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>

static std::string trim(std::string s) {
  auto a = s.find_first_not_of(" \t\r\n[]");
  auto b = s.find_last_not_of(" \t\r\n[]");
  return (a == std::string::npos) ? "" : s.substr(a, b - a + 1);
}

class MapPublisher : public rclcpp::Node {
public:
  MapPublisher() : Node("map_publisher") {
    std::string yaml = declare_parameter("map_yaml", std::string(""));
    frame_ = declare_parameter("frame_id", std::string("map"));
    rclcpp::QoS qos(1); qos.transient_local();   // latched
    pub_ = create_publisher<nav_msgs::msg::OccupancyGrid>("/map", qos);
    if (!load(yaml)) {
      RCLCPP_ERROR(get_logger(), "map_publisher: failed to load %s", yaml.c_str());
      return;
    }
    pub_->publish(grid_);
    timer_ = create_wall_timer(std::chrono::seconds(2), [this]{ pub_->publish(grid_); });
    RCLCPP_INFO(get_logger(), "map_publisher: /map %ux%u @ %.3f m/px",
                grid_.info.width, grid_.info.height, grid_.info.resolution);
  }

private:
  bool load(const std::string& yaml_path) {
    std::ifstream f(yaml_path);
    if (!f) return false;
    std::string image, line; double res = 0.05; double ox=0, oy=0;
    double occ_th = 0.65, free_th = 0.25; int negate = 0;
    while (std::getline(f, line)) {
      auto c = line.find(':'); if (c == std::string::npos) continue;
      std::string k = trim(line.substr(0, c)), v = trim(line.substr(c + 1));
      if (k == "image") image = v;
      else if (k == "resolution") res = std::stod(v);
      else if (k == "occupied_thresh") occ_th = std::stod(v);
      else if (k == "free_thresh") free_th = std::stod(v);
      else if (k == "negate") negate = std::stoi(v);
      else if (k == "origin") {
        std::stringstream ss(v); std::string t; std::vector<double> o;
        while (std::getline(ss, t, ',')) o.push_back(std::stod(trim(t)));
        if (o.size() >= 2) { ox = o[0]; oy = o[1]; }
      }
    }
    std::string dir = yaml_path.substr(0, yaml_path.find_last_of('/') + 1);
    std::string pgm = (image.front() == '/') ? image : dir + image;

    std::ifstream img(pgm, std::ios::binary);
    if (!img) return false;
    std::string magic; img >> magic;
    int W, H, maxv; img >> W >> H >> maxv; img.get();  // consume one whitespace
    std::vector<int> px(W * H);
    if (magic == "P5") {
      std::vector<unsigned char> buf(W * H);
      img.read(reinterpret_cast<char*>(buf.data()), W * H);
      for (int i = 0; i < W * H; ++i) px[i] = buf[i];
    } else {  // P2 ascii
      for (int i = 0; i < W * H; ++i) img >> px[i];
    }

    grid_.header.frame_id = frame_;
    grid_.info.resolution = res;
    grid_.info.width = W; grid_.info.height = H;
    grid_.info.origin.position.x = ox; grid_.info.origin.position.y = oy;
    grid_.info.origin.orientation.w = 1.0;
    grid_.data.assign(W * H, -1);
    // map_server convention: row 0 of OccupancyGrid is the BOTTOM (origin), so
    // flip the PGM (whose row 0 is the top) vertically.
    for (int r = 0; r < H; ++r)
      for (int col = 0; col < W; ++col) {
        int p = px[r * W + col];
        double occ = negate ? (double)p / maxv : (double)(maxv - p) / maxv;
        int8_t val = (occ >= occ_th) ? 100 : (occ <= free_th ? 0 : -1);
        grid_.data[(H - 1 - r) * W + col] = val;
      }
    return true;
  }

  std::string frame_;
  nav_msgs::msg::OccupancyGrid grid_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv){ rclcpp::init(argc,argv); rclcpp::spin(std::make_shared<MapPublisher>()); rclcpp::shutdown(); return 0; }
