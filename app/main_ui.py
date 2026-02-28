"""
道路异常事件检测系统 - PyQt5 图形界面

功能:
  1. 选择视频文件进行检测
  2. 摄像头实时检测
  3. 批量图片检测
  4. 检测结果实时显示
  5. 检测日志记录与导出
  6. 三类异常事件统计: 抛洒物 / 违停 / 逆行
  7. 支持检测参数调节 (置信度/IOU阈值)
  8. 视频级推理: TVAD 三维度聚合判定

使用方法:
    python app/main_ui.py
    python app/main_ui.py --weights path/to/best.pt
"""
import sys
import os
import cv2
import time
import numpy as np
from collections import defaultdict
from datetime import datetime

# 添加项目根目录
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)
from PyQt5.QtGui import QFont, QImage, QPixmap

# 注册自定义模块
from models.register_modules import register_custom_modules
register_custom_modules()

from ultralytics import YOLO
from models.modules.tvad import TVAD, VideoDecisionConfig, FrameDetection


# ============ 类别定义 ============
CLASS_INFO = {
    0: {'name': '抛洒物', 'en': 'debris', 'color': (0, 0, 255), 'icon': '🚧'},
    1: {'name': '机动车违停', 'en': 'illegal_parking', 'color': (0, 165, 255), 'icon': '🚗'},
    2: {'name': '逆行', 'en': 'retrograde', 'color': (255, 0, 0), 'icon': '🔄'},
}


class DetectionThread(QThread):
    """检测工作线程"""
    frame_signal = pyqtSignal(np.ndarray)
    log_signal = pyqtSignal(str)
    stats_signal = pyqtSignal(dict)
    fps_signal = pyqtSignal(float)
    finished_signal = pyqtSignal()
    
    def __init__(self, source, model_path, conf=0.5, iou=0.45):
        super().__init__()
        self.source = source
        self.model_path = model_path
        self.conf = conf
        self.iou = iou
        self.running = True
        self._stats = defaultdict(int)
    
    def run(self):
        try:
            model = YOLO(self.model_path)
            
            # 判断源类型
            if isinstance(self.source, int) or self.source.isdigit():
                source = int(self.source) if isinstance(self.source, str) else self.source
            else:
                source = self.source
            
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                self.log_signal.emit(f"[错误] 无法打开: {source}")
                return
            
            fps_counter = 0
            fps_start = time.time()
            
            while self.running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 推理
                results = model(frame, conf=self.conf, iou=self.iou, verbose=False)
                
                # 绘制结果
                frame_annotated = frame.copy()
                frame_stats = defaultdict(int)
                
                for r in results:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        info = CLASS_INFO.get(cls_id, {
                            'name': f'class_{cls_id}', 'color': (0, 255, 0), 'icon': '?'
                        })
                        
                        color = info['color']
                        label = f"{info['name']} {conf:.2f}"
                        
                        # 绘制框
                        cv2.rectangle(frame_annotated, (x1, y1), (x2, y2), color, 2)
                        
                        # 绘制标签背景
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                        cv2.rectangle(frame_annotated, (x1, y1 - th - 10), 
                                    (x1 + tw, y1), color, -1)
                        cv2.putText(frame_annotated, label, (x1, y1 - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        frame_stats[cls_id] += 1
                        self._stats[cls_id] += 1
                        
                        # 发送日志
                        timestamp = datetime.now().strftime('%H:%M:%S')
                        self.log_signal.emit(
                            f"[{timestamp}] {info['icon']} 检测到: {info['name']} "
                            f"(置信度: {conf:.2f})"
                        )
                
                # 计算 FPS
                fps_counter += 1
                if fps_counter % 10 == 0:
                    elapsed = time.time() - fps_start
                    fps = fps_counter / elapsed if elapsed > 0 else 0
                    self.fps_signal.emit(fps)
                
                self.frame_signal.emit(frame_annotated)
                self.stats_signal.emit(dict(self._stats))
                
        except Exception as e:
            self.log_signal.emit(f"[错误] {str(e)}")
        finally:
            if 'cap' in locals():
                cap.release()
            self.finished_signal.emit()
    
    def stop(self):
        self.running = False


class VideoLevelThread(QThread):
    """视频级推理工作线程 (TVAD 三维度聚合判定)"""
    log_signal = pyqtSignal(str)
    result_signal = pyqtSignal(dict)
    finished_signal = pyqtSignal()
    
    def __init__(self, video_path, model_path, conf=0.5, iou=0.45):
        super().__init__()
        self.video_path = video_path
        self.model_path = model_path
        self.conf = conf
        self.iou = iou
    
    def run(self):
        try:
            import cv2
            model = YOLO(self.model_path)
            tvad = TVAD()
            
            self.log_signal.emit(f"[TVAD] 开始视频级分析: {os.path.basename(self.video_path)}")
            
            cap = cv2.VideoCapture(self.video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            self.log_signal.emit(f"[TVAD] 视频信息: {total}帧, {fps:.1f}FPS, "
                               f"时长{total/fps:.1f}s")
            
            results = model.predict(
                source=self.video_path, conf=self.conf, iou=self.iou,
                imgsz=640, save=False, stream=True, verbose=False,
            )
            
            frame_dets = []
            frame_idx = 0
            for r in results:
                for box in r.boxes:
                    frame_dets.append(FrameDetection(
                        frame_idx=frame_idx,
                        cls_id=int(box.cls[0]),
                        confidence=float(box.conf[0]),
                    ))
                frame_idx += 1
            
            self.log_signal.emit(f"[TVAD] 逐帧推理完成, 共{frame_idx}帧, "
                               f"检出{len(frame_dets)}个目标")
            
            decision = tvad.decide(frame_dets, frame_idx, fps=fps)
            report = tvad.format_report(decision, os.path.basename(self.video_path))
            
            for line in report.split('\n'):
                self.log_signal.emit(line)
            
            result_dict = tvad.to_dict(decision)
            result_dict['video'] = os.path.basename(self.video_path)
            self.result_signal.emit(result_dict)
            
        except Exception as e:
            self.log_signal.emit(f"[TVAD 错误] {str(e)}")
        finally:
            self.finished_signal.emit()


class MainWindow(QMainWindow):
    """主窗口"""
    
    def __init__(self, model_path=None):
        super().__init__()
        self.model_path = model_path or self._find_model()
        self.det_thread = None
        self.tvad_thread = None
        self.init_ui()
    
    def _find_model(self):
        """自动查找模型权重"""
        candidates = [
            'runs/road_anomaly/yolov11m_improved/weights/best.pt',
            'runs/road_anomaly/A3_full/weights/best.pt',
            'best.pt',
            'yolo11m.pt',
        ]
        for p in candidates:
            if os.path.exists(os.path.join(PROJECT_ROOT, p)):
                return os.path.join(PROJECT_ROOT, p)
        return 'yolo11m.pt'
    
    def init_ui(self):
        self.setWindowTitle("🚦 道路异常事件检测系统 v1.0")
        self.setGeometry(100, 50, 1500, 950)
        self.setStyleSheet("""
            QMainWindow { background-color: #f5f6fa; }
            QPushButton {
                padding: 10px 20px; border-radius: 5px;
                font-size: 14px; font-weight: bold;
                border: none; color: white;
            }
            QPushButton:hover { opacity: 0.9; }
            QLabel { font-size: 13px; }
            QTextBrowser { 
                background: #2d3436; color: #dfe6e9;
                font-family: Consolas, monospace; font-size: 12px;
                border-radius: 5px;
            }
        """)
        
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(15, 10, 15, 10)
        
        # ===== 标题栏 =====
        title = QLabel("🚦 基于改进YOLOv11的多类型道路异常事件检测系统")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            "font-size: 24px; font-weight: bold; color: #2d3436; "
            "padding: 10px; background: white; border-radius: 8px;"
        )
        main_layout.addWidget(title)
        
        # ===== 主内容区 =====
        content = QHBoxLayout()
        
        # 左侧: 视频显示
        left_panel = QVBoxLayout()
        
        self.video_label = QLabel("请选择视频文件或开启摄像头")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(960, 540)
        self.video_label.setStyleSheet(
            "border: 2px solid #b2bec3; background: #dfe6e9; "
            "border-radius: 8px; font-size: 16px; color: #636e72;"
        )
        left_panel.addWidget(self.video_label)
        
        # FPS 显示
        self.fps_label = QLabel("FPS: --")
        self.fps_label.setStyleSheet(
            "font-size: 14px; color: #0984e3; font-weight: bold;"
        )
        left_panel.addWidget(self.fps_label)
        
        content.addLayout(left_panel, 3)
        
        # 右侧: 控制面板 + 统计
        right_panel = QVBoxLayout()
        
        # 模型信息
        model_label = QLabel(f"📦 模型: {os.path.basename(self.model_path)}")
        model_label.setStyleSheet(
            "padding: 8px; background: white; border-radius: 5px; "
            "font-size: 12px;"
        )
        right_panel.addWidget(model_label)
        
        # 参数调节
        param_group = QGroupBox("参数设置")
        param_layout = QFormLayout()
        
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(10, 95)
        self.conf_slider.setValue(50)
        self.conf_label = QLabel("0.50")
        self.conf_slider.valueChanged.connect(
            lambda v: self.conf_label.setText(f"{v/100:.2f}"))
        conf_row = QHBoxLayout()
        conf_row.addWidget(self.conf_slider)
        conf_row.addWidget(self.conf_label)
        param_layout.addRow("置信度:", conf_row)
        
        self.iou_slider = QSlider(Qt.Horizontal)
        self.iou_slider.setRange(20, 80)
        self.iou_slider.setValue(45)
        self.iou_label = QLabel("0.45")
        self.iou_slider.valueChanged.connect(
            lambda v: self.iou_label.setText(f"{v/100:.2f}"))
        iou_row = QHBoxLayout()
        iou_row.addWidget(self.iou_slider)
        iou_row.addWidget(self.iou_label)
        param_layout.addRow("IOU:", iou_row)
        
        param_group.setLayout(param_layout)
        right_panel.addWidget(param_group)
        
        # 统计面板
        stats_group = QGroupBox("检测统计")
        stats_layout = QVBoxLayout()
        
        self.stats_labels = {}
        for cls_id, info in CLASS_INFO.items():
            label = QLabel(f"{info['icon']} {info['name']}: 0")
            label.setStyleSheet(
                f"font-size: 16px; font-weight: bold; padding: 5px; "
                f"color: rgb({info['color'][2]},{info['color'][1]},{info['color'][0]});"
            )
            stats_layout.addWidget(label)
            self.stats_labels[cls_id] = label
        
        self.total_label = QLabel("📊 总计: 0")
        self.total_label.setStyleSheet(
            "font-size: 16px; font-weight: bold; padding: 5px; color: #2d3436;"
        )
        stats_layout.addWidget(self.total_label)
        
        stats_group.setLayout(stats_layout)
        right_panel.addWidget(stats_group)
        
        right_panel.addStretch()
        content.addLayout(right_panel, 1)
        main_layout.addLayout(content)
        
        # ===== 按钮栏 =====
        btn_layout = QHBoxLayout()
        
        self.btn_file = QPushButton("📁 选择视频")
        self.btn_file.setStyleSheet("background-color: #0984e3;")
        self.btn_file.clicked.connect(self.open_file)
        btn_layout.addWidget(self.btn_file)
        
        self.btn_image = QPushButton("🖼 选择图片")
        self.btn_image.setStyleSheet("background-color: #6c5ce7;")
        self.btn_image.clicked.connect(self.open_image)
        btn_layout.addWidget(self.btn_image)
        
        self.btn_camera = QPushButton("📹 摄像头")
        self.btn_camera.setStyleSheet("background-color: #00b894;")
        self.btn_camera.clicked.connect(self.start_camera)
        btn_layout.addWidget(self.btn_camera)
        
        self.btn_tvad = QPushButton("🎬 视频级判定")
        self.btn_tvad.setStyleSheet("background-color: #fdcb6e;")
        self.btn_tvad.clicked.connect(self.start_video_level)
        btn_layout.addWidget(self.btn_tvad)
        
        self.btn_stop = QPushButton("⏹ 停止")
        self.btn_stop.setStyleSheet("background-color: #e17055;")
        self.btn_stop.clicked.connect(self.stop_detection)
        self.btn_stop.setEnabled(False)
        btn_layout.addWidget(self.btn_stop)
        
        self.btn_model = QPushButton("🔧 切换模型")
        self.btn_model.setStyleSheet("background-color: #636e72;")
        self.btn_model.clicked.connect(self.change_model)
        btn_layout.addWidget(self.btn_model)
        
        main_layout.addLayout(btn_layout)
        
        # ===== 日志区 =====
        self.log_browser = QTextBrowser()
        self.log_browser.setMaximumHeight(180)
        self.log_browser.setPlaceholderText("检测日志...")
        main_layout.addWidget(self.log_browser)
    
    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "",
            "视频文件 (*.mp4 *.avi *.mkv *.mov *.flv);;所有文件 (*.*)")
        if path:
            self.start_detection(path)
    
    def open_image(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "选择图片", "",
            "图片文件 (*.jpg *.jpeg *.png *.bmp);;所有文件 (*.*)")
        if paths:
            # 对图片逐个推理
            model = YOLO(self.model_path)
            conf = self.conf_slider.value() / 100
            iou = self.iou_slider.value() / 100
            
            for p in paths:
                results = model(p, conf=conf, iou=iou, verbose=False)
                for r in results:
                    annotated = r.plot()
                    self.display_frame(annotated)
                    
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        c = float(box.conf[0])
                        info = CLASS_INFO.get(cls_id, {'name': '未知', 'icon': '?'})
                        self.log_browser.append(
                            f"[图片] {info['icon']} {info['name']} "
                            f"(置信度: {c:.2f}) - {os.path.basename(p)}")
    
    def start_camera(self):
        self.start_detection(0)
    
    def start_detection(self, source):
        self.stop_detection()
        
        conf = self.conf_slider.value() / 100
        iou = self.iou_slider.value() / 100
        
        self.det_thread = DetectionThread(source, self.model_path, conf, iou)
        self.det_thread.frame_signal.connect(self.display_frame)
        self.det_thread.log_signal.connect(self.log_browser.append)
        self.det_thread.stats_signal.connect(self.update_stats)
        self.det_thread.fps_signal.connect(self.update_fps)
        self.det_thread.finished_signal.connect(self.on_detection_finished)
        self.det_thread.start()
        
        self.btn_stop.setEnabled(True)
        self.btn_file.setEnabled(False)
        self.btn_camera.setEnabled(False)
        
        self.log_browser.append(f"[系统] 开始检测: {source}")
    
    def stop_detection(self):
        if self.det_thread and self.det_thread.isRunning():
            self.det_thread.stop()
            self.det_thread.wait(3000)
            self.log_browser.append("[系统] 检测已停止")
        self.on_detection_finished()
    
    def on_detection_finished(self):
        self.btn_stop.setEnabled(False)
        self.btn_file.setEnabled(True)
        self.btn_camera.setEnabled(True)
    
    def display_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(img).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(pixmap)
    
    def update_stats(self, stats: dict):
        total = 0
        for cls_id, label in self.stats_labels.items():
            count = stats.get(cls_id, 0)
            info = CLASS_INFO[cls_id]
            label.setText(f"{info['icon']} {info['name']}: {count}")
            total += count
        self.total_label.setText(f"📊 总计: {total}")
    
    def update_fps(self, fps: float):
        self.fps_label.setText(f"FPS: {fps:.1f}")
    
    def change_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择模型权重", "",
            "权重文件 (*.pt *.onnx);;所有文件 (*.*)")
        if path:
            self.model_path = path
            self.log_browser.append(f"[系统] 模型已切换: {os.path.basename(path)}")
    
    def start_video_level(self):
        """启动视频级推理 (TVAD)"""
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件进行视频级判定", "",
            "视频文件 (*.mp4 *.avi *.mkv *.mov *.flv);;所有文件 (*.*)")
        if not path:
            return
        
        conf = self.conf_slider.value() / 100
        iou = self.iou_slider.value() / 100
        
        self.tvad_thread = VideoLevelThread(path, self.model_path, conf, iou)
        self.tvad_thread.log_signal.connect(self.log_browser.append)
        self.tvad_thread.result_signal.connect(self.on_tvad_result)
        self.tvad_thread.finished_signal.connect(
            lambda: self.btn_tvad.setEnabled(True))
        
        self.btn_tvad.setEnabled(False)
        self.tvad_thread.start()
        self.log_browser.append(f"[系统] 启动 TVAD 视频级判定: {os.path.basename(path)}")
    
    def on_tvad_result(self, result: dict):
        """处理 TVAD 判定结果"""
        pe = result.get('primary_event')
        vname = result.get('video', '')
        
        if pe:
            msg = (f"视频: {vname}\n\n"
                   f"★ 主事件: {pe['name_cn']}\n"
                   f"综合分数: {pe['score']:.6f}\n"
                   f"帧占比: {pe['frame_ratio']:.4f}\n"
                   f"时序一致性: {pe['temporal_consistency']:.4f}\n"
                   f"平均置信度: {pe['avg_confidence']:.4f}")
            QMessageBox.information(self, "TVAD 视频级判定结果", msg)
        else:
            QMessageBox.information(
                self, "TVAD 视频级判定结果",
                f"视频: {vname}\n\n未检出异常事件")
    
    def closeEvent(self, event):
        self.stop_detection()
        if self.tvad_thread and self.tvad_thread.isRunning():
            self.tvad_thread.wait(3000)
        event.accept()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=None)
    args = parser.parse_args()
    
    app = QApplication(sys.argv)
    
    # 设置全局字体
    font = QFont()
    font.setFamily("Microsoft YaHei")
    font.setPointSize(10)
    app.setFont(font)
    
    window = MainWindow(model_path=args.weights)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
