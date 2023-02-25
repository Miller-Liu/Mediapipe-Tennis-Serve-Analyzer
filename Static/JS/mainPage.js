const getFile1 = document.getElementById("uploadVideo1")
const FileButton1 = document.getElementById("InputButton1")
const getFile2 = document.getElementById("uploadVideo2")
const FileButton2 = document.getElementById("InputButton2")
const inputBackground1 = document.getElementById("InputGroup1Background")
const inputBackground2 = document.getElementById("InputGroup2Background")
const videoIcons = document.getElementsByClassName("video-icon");
const FileLogo1 = document.getElementById("video-icon-1");
const FileLogo2 = document.getElementById("video-icon-2");
const UploadDiv = document.getElementById("UploadButton");
const UploadButton = document.getElementById("UploadButtonText");
const mainDiv = document.getElementById("MainDiv");
const loading = document.getElementById("LoadingPage");
const loader_collection = Array.from(document.getElementsByClassName('Loader'));
const processed = document.getElementById("ProcessedPage");
const carousel = document.getElementById("carouselSlides");
var media1 = null;
var media2 = null;
var file1name = '';
var file2name = '';

FileButton1.addEventListener("click", function() {
    getFile1.click();
});

FileButton2.addEventListener("click", function() {
    getFile2.click();
});

UploadButton.addEventListener("click", function() {
    mainDiv.style.transition = '1s ease-in';
    mainDiv.style.top = "-100vh"
    loading.style.transition = '1s ease-in';
    loading.style.top = '0';
    setTimeout(() => {
        loading.style.opacity = '1';
        loader_collection.forEach((loader, i) => {
            setTimeout(() => {
                loader.style.animation = 'loading 3s linear infinite';
            }, 500*i);
        });
    }, 1000);
    var fd = new FormData();
    fd.append("video1", $('#uploadVideo1').prop('files')[0]);
    fd.append("video2", $('#uploadVideo2').prop('files')[0]);
    $.ajax({
        url: '/upload',
        type: 'post',
        data: fd,
        contentType: false,
        processData: false,
        success: function(response){
            console.log("Uploaded");
        }
    });
    $.ajax({
        url: '/process',
        type: 'get',
        contentType: false,
        processData: false,
        success: function(response){
            console.log("Processed")
            loading.style.transition = '1s ease-in';
            loading.style.top = '-100vh';
            processed.style.transition = '1s ease-in';
            processed.style.top = '0';
            $("#vid1").attr("src","Static/uploads/video1.mp4");
            $("#vid2").attr("src","Static/uploads/video2.mp4");
            $("#vid3").attr("src","Static/Videos/video1.mp4");
            $("#vid4").attr("src","Static/Videos/video2.mp4");
        }
    });
});

function FileButton1Hover() {
    inputBackground1.style.background = '#C1C1C1';
    inputBackground1.style.border = 'solid #C1C1C1';
};

function FileButton1Leave() {
    inputBackground1.style.background = '#CCCCCC';
    inputBackground1.style.border = 'solid #CCCCCC';
};

function FileButton2Hover() {
    inputBackground2.style.background = '#C1C1C1';
    inputBackground2.style.border = 'solid #C1C1C1';
};

function FileButton2Leave() {
    inputBackground2.style.background = '#CCCCCC';
    inputBackground2.style.border = 'solid #CCCCCC';
};

getFile1.addEventListener("change", function() {
    if (getFile1.value) {
        media1 = URL.createObjectURL(getFile1.files[0]);
        FileButton1.innerHTML = "Uploaded " + getFile1.files[0].name;
        FileLogo1.src = "Static/Website-Images/Check-Mark-Icon.png";
        FileLogo1.style.height = '110px';
    }
    else {
        media1 = null
        FileButton1.innerHTML = "CHOOSE A FILE";
        FileLogo1.src = "Static/Website-Images/Video-Icon.png";
        FileLogo1.style.height = '';
    }
    if (media1 && media2) {
        UploadDiv.style.display = "flex";
    }
    else {
        UploadDiv.style.display = "none";
    }
})

getFile2.addEventListener("change", function() {
    if (getFile2.value) {
        media2 = URL.createObjectURL(getFile2.files[0]);
        FileButton2.innerHTML = "Uploaded " + getFile2.files[0].name;
        FileLogo2.src = "Static/Website-Images/Check-Mark-Icon.png";
        FileLogo2.style.height = '110px';
    }
    else {
        media2 = null
        FileButton2.innerHTML = "CHOOSE A FILE";
        FileLogo2.src = "Static/Website-Images/Video-Icon.png";
        FileLogo2.style.height = '';
    }
    if (media1 && media2) {
        UploadDiv.style.display = "flex";
    }
    else {
        UploadDiv.style.display = "none";
    }
})

function PlayBothVids() {
    $("#vid1").get(0).play()
    $("#vid2").get(0).play()
    $("#vid3").get(0).play()
    $("#vid4").get(0).play()
}
