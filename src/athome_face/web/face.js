(function () {
  "use strict";

  // Default capsule eye
  const PATH_CAPSULE = "M 30,50 C 30,20 130,20 130,50 L 130,170 C 130,200 30,200 30,170 Z";
  // Happy crescent eye (^_^)
  const PATH_HAPPY = "M 30,140 Q 80,45 130,140 Q 80,85 30,140 Z";
  // Focused / Sharp eye for following & grabbing
  const PATH_FOCUSED = "M 30,70 C 40,40 120,40 130,70 L 130,165 C 120,195 40,195 30,165 Z";
  // Wide open surprised / curious / lost eye
  const PATH_WIDE = "M 20,40 C 20,10 140,10 140,40 L 140,180 C 140,210 20,210 20,180 Z";

  const leftEyeShape = document.getElementById("left-eye-shape");
  const rightEyeShape = document.getElementById("right-eye-shape");
  const leftEyeGroup = document.getElementById("left-eye-group");
  const rightEyeGroup = document.getElementById("right-eye-group");
  const leftPupil = document.getElementById("left-pupil");
  const rightPupil = document.getElementById("right-pupil");
  const leftStar = document.getElementById("left-star");
  const rightStar = document.getElementById("right-star");
  const leftScanline = document.getElementById("left-scanline");
  const rightScanline = document.getElementById("right-scanline");
  const leftBrow = document.getElementById("eyebrow-left");
  const rightBrow = document.getElementById("eyebrow-right");
  const leftBlush = document.getElementById("blush-left");
  const rightBlush = document.getElementById("blush-right");
  const fxBadge = document.getElementById("fx-badge");
  const mouthWave = document.getElementById("mouth-wave");
  const mouthSmile = document.getElementById("mouth-smile");
  const speechBubble = document.getElementById("speech-bubble");
  const speechText = document.getElementById("speech-text");
  const emotionLabel = document.getElementById("current-emotion-label");
  const testToolbar = document.getElementById("test-toolbar");
  const btnToggleToolbar = document.getElementById("btn-toggle-toolbar");
  const btnCloseToolbar = document.getElementById("btn-close-toolbar");
  const btnFullscreen = document.getElementById("btn-fullscreen");
  const wsDot = document.getElementById("ws-indicator");
  const wsLabel = document.getElementById("ws-label");

  const leftEyelidTop = document.getElementById("left-eyelid-top");
  const leftEyelidBottom = document.getElementById("left-eyelid-bottom");
  const rightEyelidTop = document.getElementById("right-eyelid-top");
  const rightEyelidBottom = document.getElementById("right-eyelid-bottom");

  let currentEmotion = "IDLE";
  let isBlinking = false;
  let blinkTimer = null;
  let saccadeTimer = null;
  let scanTimer = null;
  let ws = null;
  let isSpeechActive = false;

  function triggerBlink(durationMs = 140, onDone = null) {
    if (isBlinking) return;
    isBlinking = true;

    leftEyelidTop.setAttribute("height", "110");
    leftEyelidBottom.setAttribute("y", "110");
    leftEyelidBottom.setAttribute("height", "110");

    rightEyelidTop.setAttribute("height", "110");
    rightEyelidBottom.setAttribute("y", "110");
    rightEyelidBottom.setAttribute("height", "110");

    setTimeout(() => {
      leftEyelidTop.setAttribute("height", "0");
      leftEyelidBottom.setAttribute("y", "220");
      leftEyelidBottom.setAttribute("height", "0");

      rightEyelidTop.setAttribute("height", "0");
      rightEyelidBottom.setAttribute("y", "220");
      rightEyelidBottom.setAttribute("height", "0");

      setTimeout(() => {
        isBlinking = false;
        if (onDone) onDone();
      }, 50);
    }, durationMs);
  }

  function startRandomBlinking() {
    clearTimeout(blinkTimer);
    const nextBlinkMs = 2800 + Math.random() * 3500;
    blinkTimer = setTimeout(() => {
      const isDouble = Math.random() < 0.2;
      triggerBlink(130, () => {
        if (isDouble) {
          setTimeout(() => {
            triggerBlink(110, startRandomBlinking);
          }, 90);
        } else {
          startRandomBlinking();
        }
      });
    }, nextBlinkMs);
  }

  function startRandomSaccades() {
    clearTimeout(saccadeTimer);
    if (currentEmotion !== "IDLE") return;

    const nextGlanceMs = 2500 + Math.random() * 3000;
    saccadeTimer = setTimeout(() => {
      if (currentEmotion === "IDLE") {
        const dx = (Math.random() - 0.5) * 24;
        const dy = (Math.random() - 0.5) * 16;
        movePupils(dx, dy);

        setTimeout(() => {
          if (currentEmotion === "IDLE") movePupils(0, 0);
        }, 1100);
      }
      startRandomSaccades();
    }, nextGlanceMs);
  }

  function movePupils(deltaX, deltaY) {
    const baseX = 80;
    const baseY = 110;
    leftPupil.setAttribute("cx", baseX + deltaX);
    leftPupil.setAttribute("cy", baseY + deltaY);
    rightPupil.setAttribute("cx", baseX + deltaX);
    rightPupil.setAttribute("cy", baseY + deltaY);
  }

  function resetAllDecorations() {
    clearInterval(scanTimer);
    leftScanline.classList.remove("active");
    rightScanline.classList.remove("active");
    leftScanline.setAttribute("opacity", "0");
    rightScanline.setAttribute("opacity", "0");

    leftStar.setAttribute("opacity", "0");
    rightStar.setAttribute("opacity", "0");

    leftBlush.classList.remove("active");
    rightBlush.classList.remove("active");

    fxBadge.className = "fx-badge";
    fxBadge.innerText = "";

    mouthWave.classList.remove("active");
    mouthSmile.classList.remove("active");

    leftPupil.setAttribute("rx", "35");
    leftPupil.setAttribute("ry", "45");
    rightPupil.setAttribute("rx", "35");
    rightPupil.setAttribute("ry", "45");
    movePupils(0, 0);

    document.getElementById("eye-left").style.transform = "none";
    document.getElementById("eye-right").style.transform = "none";

    leftBrow.style.transform = "none";
    rightBrow.style.transform = "none";
    leftBrow.style.background = "#00f2ff";
    rightBrow.style.background = "#00f2ff";

    leftEyeShape.setAttribute("fill", "url(#grad-cyan)");
    rightEyeShape.setAttribute("fill", "url(#grad-cyan)");
  }

  function setEmotion(emotion) {
    currentEmotion = emotion.toUpperCase();
    emotionLabel.innerText = currentEmotion;
    resetAllDecorations();

    document.querySelectorAll(".btn-expr").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.expr === currentEmotion);
    });

    switch (currentEmotion) {
      case "IDLE":
      default:
        leftEyeShape.setAttribute("d", PATH_CAPSULE);
        rightEyeShape.setAttribute("d", PATH_CAPSULE);
        startRandomSaccades();
        break;

      case "SEARCHING":
      case "SCAN":
        leftEyeShape.setAttribute("d", PATH_FOCUSED);
        rightEyeShape.setAttribute("d", PATH_FOCUSED);
        leftScanline.classList.add("active");
        rightScanline.classList.add("active");
        leftScanline.setAttribute("opacity", "0.85");
        rightScanline.setAttribute("opacity", "0.85");

        let scanDir = 1;
        scanTimer = setInterval(() => {
          movePupils(scanDir * 28, 0);
          scanDir = -scanDir;
        }, 850);
        break;

      case "GREETING":
      case "WELCOME":
      case "HAPPY":
        leftEyeShape.setAttribute("d", PATH_HAPPY);
        rightEyeShape.setAttribute("d", PATH_HAPPY);
        leftEyeShape.setAttribute("fill", "url(#grad-happy)");
        rightEyeShape.setAttribute("fill", "url(#grad-happy)");
        leftBlush.classList.add("active");
        rightBlush.classList.add("active");
        mouthSmile.classList.add("active");
        leftBrow.style.transform = "translateY(-12px) rotate(-6deg)";
        rightBrow.style.transform = "translateY(-12px) rotate(6deg)";
        break;

      case "LISTENING":
        leftEyeShape.setAttribute("d", PATH_WIDE);
        rightEyeShape.setAttribute("d", PATH_WIDE);
        leftPupil.setAttribute("rx", "45");
        leftPupil.setAttribute("ry", "55");
        rightPupil.setAttribute("rx", "45");
        rightPupil.setAttribute("ry", "55");
        mouthWave.classList.add("active");
        leftBrow.style.transform = "translateY(-10px)";
        rightBrow.style.transform = "translateY(-10px)";
        break;

      case "SPEAKING":
        leftEyeShape.setAttribute("d", PATH_CAPSULE);
        rightEyeShape.setAttribute("d", PATH_CAPSULE);
        mouthWave.classList.add("active");
        break;

      case "FOLLOWING":
      case "FOCUSED":
        leftEyeShape.setAttribute("d", PATH_FOCUSED);
        rightEyeShape.setAttribute("d", PATH_FOCUSED);
        document.getElementById("eye-left").style.transform = "rotate(4deg)";
        document.getElementById("eye-right").style.transform = "rotate(-4deg)";
        leftBrow.style.transform = "translateY(5px) rotate(8deg)";
        rightBrow.style.transform = "translateY(5px) rotate(-8deg)";
        break;

      case "CURIOUS":
      case "WAIT_GESTURE":
        leftEyeShape.setAttribute("d", PATH_WIDE);
        rightEyeShape.setAttribute("d", PATH_CAPSULE);
        leftBrow.style.transform = "translateY(-22px) rotate(-14deg)";
        rightBrow.style.transform = "translateY(6px) rotate(-8deg)";
        movePupils(-12, -10);
        fxBadge.innerText = "?";
        fxBadge.classList.add("show-question");
        break;

      case "GRABBING":
      case "WORKING":
        leftEyeShape.setAttribute("d", PATH_FOCUSED);
        rightEyeShape.setAttribute("d", PATH_FOCUSED);
        leftEyeShape.setAttribute("fill", "url(#grad-cyan)");
        rightEyeShape.setAttribute("fill", "url(#grad-cyan)");
        movePupils(0, 28);
        leftBrow.style.transform = "translateY(10px) rotate(10deg)";
        rightBrow.style.transform = "translateY(10px) rotate(-10deg)";
        break;

      case "SUCCESS":
      case "DONE":
      case "WIN":
        leftEyeShape.setAttribute("d", PATH_HAPPY);
        rightEyeShape.setAttribute("d", PATH_HAPPY);
        leftEyeShape.setAttribute("fill", "url(#grad-gold)");
        rightEyeShape.setAttribute("fill", "url(#grad-gold)");
        leftStar.setAttribute("opacity", "1");
        rightStar.setAttribute("opacity", "1");
        mouthSmile.classList.add("active");
        leftBrow.style.transform = "translateY(-16px) rotate(-8deg)";
        rightBrow.style.transform = "translateY(-16px) rotate(8deg)";
        leftBrow.style.background = "#ffd700";
        rightBrow.style.background = "#ffd700";
        break;

      case "LOST":
      case "WARNING":
      case "ALERT":
        leftEyeShape.setAttribute("d", PATH_WIDE);
        rightEyeShape.setAttribute("d", PATH_WIDE);
        leftEyeShape.setAttribute("fill", "url(#grad-warning)");
        rightEyeShape.setAttribute("fill", "url(#grad-warning)");
        leftBrow.style.transform = "translateY(-12px) rotate(16deg)";
        rightBrow.style.transform = "translateY(-12px) rotate(-16deg)";
        leftBrow.style.background = "#ff3344";
        rightBrow.style.background = "#ff3344";
        fxBadge.innerText = "!";
        fxBadge.classList.add("show-warning");
        break;
    }
  }

  const stage = document.getElementById("face-stage");

  stage.addEventListener("pointerdown", (e) => {
    if (e.target.closest("#hud-bar") || e.target.closest("#test-toolbar")) return;

    const rect = stage.getBoundingClientRect();
    const x = e.clientX - rect.left - rect.width / 2;
    const y = e.clientY - rect.top - rect.height / 2;

    const clampX = Math.max(-30, Math.min(30, x * 0.12));
    const clampY = Math.max(-25, Math.min(25, y * 0.12));
    movePupils(clampX, clampY);

    if (Math.abs(x) > 100 && y > 30) {
      triggerCuteWink();
    }
  });

  stage.addEventListener("pointerup", () => {
    setTimeout(() => {
      if (currentEmotion === "IDLE") movePupils(0, 0);
    }, 800);
  });

  function triggerCuteWink() {
    rightEyelidTop.setAttribute("height", "110");
    rightEyelidBottom.setAttribute("y", "110");
    rightEyelidBottom.setAttribute("height", "110");
    leftBlush.classList.add("active");
    rightBlush.classList.add("active");
    mouthSmile.classList.add("active");

    setTimeout(() => {
      rightEyelidTop.setAttribute("height", "0");
      rightEyelidBottom.setAttribute("y", "220");
      rightEyelidBottom.setAttribute("height", "0");
      setTimeout(() => {
        if (currentEmotion !== "GREETING") {
          leftBlush.classList.remove("active");
          rightBlush.classList.remove("active");
          mouthSmile.classList.remove("active");
        }
      }, 1000);
    }, 450);
  }

  function connectWebSocket() {
    const host = window.location.hostname || "localhost";
    const wsUrl = `ws://${host}:8766`;

    try {
      ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        wsDot.className = "dot connected";
        wsLabel.innerText = "ROS 2 CONNECTED";
        console.log("[AthomeFace] Connected to ROS 2 WebSocket on " + wsUrl);
      };

      ws.onclose = () => {
        wsDot.className = "dot disconnected";
        wsLabel.innerText = "STANDALONE PREVIEW";
        setTimeout(connectWebSocket, 3000);
      };

      ws.onerror = () => {
        wsDot.className = "dot disconnected";
        wsLabel.innerText = "STANDALONE PREVIEW";
      };

      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);

          if (msg.expression) {
            setEmotion(msg.expression);
          }
          if (msg.state) {
            handleRobotStateMapping(msg.state);
          }
          if (msg.type === "TTS_PLAY" || msg.type === "SPEAK") {
            setEmotion("SPEAKING");
            if (msg.text) showSubtitle(msg.text);
          } else if (msg.type === "TTS_END") {
            setEmotion("IDLE");
            hideSubtitle();
          }
          if (msg.tracking_status === "LOST") {
            setEmotion("LOST");
          }
        } catch (err) {
          console.warn("[AthomeFace] Malformed message:", err);
        }
      };
    } catch (e) {
      console.log("[AthomeFace] Running in standalone preview mode.");
    }
  }

  function handleRobotStateMapping(stateName) {
    const mapping = {
      IDLE: "IDLE",
      READY_TO_FOLLOW: "GREETING",
      FOLLOWING: "FOLLOWING",
      WAIT_GESTURE: "CURIOUS",
      ALIGNING_BAG: "GRABBING",
      GRABBING_BAG: "GRABBING",
      HOMING: "FOLLOWING",
      DONE: "SUCCESS",
    };
    const targetExpr = mapping[stateName] || "IDLE";
    setEmotion(targetExpr);
  }

  function showSubtitle(text) {
    speechText.innerText = text;
    speechBubble.classList.add("active");
  }

  function hideSubtitle() {
    speechBubble.classList.remove("active");
  }

  function toggleFullscreen() {
    if (!document.fullscreenElement && !document.webkitFullscreenElement) {
      if (document.documentElement.requestFullscreen) {
        document.documentElement.requestFullscreen().catch(() => {});
      } else if (document.documentElement.webkitRequestFullscreen) {
        document.documentElement.webkitRequestFullscreen();
      }
    } else {
      if (document.exitFullscreen) {
        document.exitFullscreen().catch(() => {});
      } else if (document.webkitExitFullscreen) {
        document.webkitExitFullscreen();
      }
    }
  }

  if (btnFullscreen) {
    btnFullscreen.addEventListener("click", (e) => {
      e.stopPropagation();
      toggleFullscreen();
    });
  }

  stage.addEventListener("dblclick", (e) => {
    if (e.target.closest("#hud-bar") || e.target.closest("#test-toolbar")) return;
    toggleFullscreen();
  });

  function toggleToolbar() {
    testToolbar.classList.toggle("hidden");
  }

  btnToggleToolbar.addEventListener("click", (e) => {
    e.stopPropagation();
    toggleToolbar();
  });

  btnCloseToolbar.addEventListener("click", (e) => {
    e.stopPropagation();
    testToolbar.classList.add("hidden");
  });

  document.querySelectorAll(".btn-expr").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      setEmotion(btn.dataset.expr);
    });
  });

  document.getElementById("btn-action-blink").addEventListener("click", (e) => {
    e.stopPropagation();
    triggerBlink(120);
  });

  document.getElementById("btn-action-touch").addEventListener("click", (e) => {
    e.stopPropagation();
    triggerCuteWink();
  });

  window.addEventListener("keydown", (e) => {
    const key = e.key.toUpperCase();
    switch (key) {
      case "1": setEmotion("IDLE"); break;
      case "2": setEmotion("SEARCHING"); break;
      case "3": setEmotion("GREETING"); break;
      case "4": setEmotion("LISTENING"); break;
      case "5": setEmotion("SPEAKING"); break;
      case "6": setEmotion("FOLLOWING"); break;
      case "7": setEmotion("CURIOUS"); break;
      case "8": setEmotion("GRABBING"); break;
      case "9": setEmotion("SUCCESS"); break;
      case "0": setEmotion("LOST"); break;
      case " ": triggerBlink(120); e.preventDefault(); break;
      case "T": toggleToolbar(); break;
      case "F": toggleFullscreen(); break;
      case "W": triggerCuteWink(); break;
    }
  });

  window.addEventListener("load", () => {
    setEmotion("IDLE");
    startRandomBlinking();
    connectWebSocket();
  });
})();

