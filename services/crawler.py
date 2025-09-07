# crawler.py
import time, tempfile, shutil, uuid, random
from typing import Dict, Any
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    NoAlertPresentException, TimeoutException, WebDriverException
)
from selenium.webdriver.common.by import By
from selenium import webdriver as wb

MAX_PAGELOAD_SEC = 25   # 페이지 로드(네트워크) 상한
MAX_WAIT_SEC = 10       # 요소 대기 상한
RETRY = 2               # 재시도 횟수 (총 1 + 2 = 3번 시도)

def crawl_recipe_stub(url: str) -> Dict[str, Any]:
    def _once() -> Dict[str, Any]:
        tmp_dir = tempfile.mkdtemp(prefix=f"chrome-profile-{uuid.uuid4()}-")
        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument(f"--user-data-dir={tmp_dir}")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1280,2000")
        # ✅ 프록시/로컬 캡처툴 완전 무시
        options.add_argument("--proxy-server=direct://")
        options.add_argument("--proxy-bypass-list=*")

        driver = None
        try:
            driver = wb.Chrome(options=options)
            driver.set_page_load_timeout(MAX_PAGELOAD_SEC)
            try:
                driver.get(url)
            except TimeoutException:
                # 전체 onload 기다리다 막히면 JS로 강제 중단 후 진행
                driver.execute_script("window.stop();")

            # 경고창 뜨면 닫기
            try:
                alert = driver.switch_to.alert
                alert.accept()
            except NoAlertPresentException:
                pass

            wait = WebDriverWait(driver, MAX_WAIT_SEC)

            def safe_text(by, selector, default=""):
                try:
                    el = wait.until(EC.presence_of_element_located((by, selector)))
                    return el.text.strip()
                except Exception:
                    return default

            def safe_attr(by, selector, attr, default=""):
                try:
                    el = wait.until(EC.presence_of_element_located((by, selector)))
                    return el.get_attribute(attr) or default
                except Exception:
                    return default

            mainImg = safe_attr(By.CSS_SELECTOR, "#main_thumbs", "src", "")
            writer  = safe_text(By.CLASS_NAME, "user_info2_name", "")
            title   = safe_text(By.CSS_SELECTOR, "#contents_area_full > div.view2_summary.st3 > h3", "")
            summary = safe_text(By.CSS_SELECTOR, "#recipeIntro", "")
            portion = safe_text(By.CLASS_NAME, "view2_summary_info1", "")
            cook_time = safe_text(By.CLASS_NAME, "view2_summary_info2", "")
            level   = safe_text(By.CLASS_NAME, "view2_summary_info3", "")

            def list_pairs(sel):
                out = []
                try:
                    items = driver.find_elements(By.CSS_SELECTOR, sel)
                    for li in items:
                        parts = li.text.split("\n")
                        out.append(parts)
                except Exception:
                    pass
                return out

            ingredient_list = list_pairs("#divConfirmedMaterialArea > ul:nth-child(1) > li")
            sauce_list      = list_pairs("#divConfirmedMaterialArea > ul:nth-child(2) > li")

            ingredient = {" ".join(x[0]): (x[1] if len(x) > 1 else "") for x in ingredient_list if x}
            sauce      = {" ".join(x[0]): (x[1] if len(x) > 1 else "") for x in sauce_list if x}

            knowHow = list({
                a.get_attribute("href")
                for a in driver.find_elements(By.CSS_SELECTOR, ".swiper-slide > a")
                if a.get_attribute("href")
            })

            step_nodes = driver.find_elements(By.CSS_SELECTOR, ".view_step_cont.media")
            step_list = []
            for s in step_nodes:
                text = s.text.replace("\n", "")
                img  = s.find_element(By.TAG_NAME, "img").get_attribute("src") if s.find_elements(By.TAG_NAME, "img") else 0
                step_list.append({text: img})

            tip = safe_text(By.CSS_SELECTOR, "#obx_recipe_step_start > dl > dd", "")

            return {
                "mainImg": mainImg,
                "writer": writer,
                "title": title,
                "summary": summary,
                "portion": portion,
                "time": cook_time,
                "level": level,
                "ingredient": ingredient,
                "sauce": sauce,
                "knowHow": knowHow,
                "step_list": step_list,
                "tip": tip,
            }

        finally:
            try:
                if driver:
                    driver.quit()
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

    # ✅ 짧은 백오프 재시도 (네트워크 흔들릴 때 대비)
    last_exc = None
    for i in range(1 + RETRY):
        try:
            return _once()
        except (TimeoutException, WebDriverException) as e:
            last_exc = e
            time.sleep(0.7 * (i + 1) + random.random() * 0.3)
    raise RuntimeError(f"Crawl failed after retries: {last_exc}")
