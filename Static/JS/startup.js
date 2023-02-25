let text1 = document.querySelector('#WelcomeText')
let text2 = document.querySelector('#Author')
let page = document.querySelector('#WelcomePage')
let container = document.querySelector('#WelcomeContainer')
let description = document.querySelector('#DescriptionContainer')
let descriptionTitle = document.querySelector('#DescriptionTitle')
let descriptionText = document.querySelector('#Description')
let skip = document.querySelector('#SkipIcon')
let skipContainer = document.querySelector('#SkipIconContainer')
let main = document.querySelector('#MainPage')
var skipIntro = false

window.addEventListener('DOMContentLoaded', ()=>{
    setTimeout(()=>{
        if (skipIntro == false) {
            container.classList.add('animation1');
        }
    }, 500)

    setTimeout(()=>{
        if (skipIntro == false) {
            container.classList.add('animation2');
        }
    }, 1000)

    setTimeout(()=>{
        if (skipIntro == false) {
            text1.classList.add('active');
            text2.classList.add('active');
        }
    }, 1500)

    setTimeout(()=>{
        if (skipIntro == false) {
            text1.classList.remove('active');
            text2.classList.remove('active');
            text1.classList.add('fade');
            text2.classList.add('fade');
        }
    }, 4500)

    setTimeout(()=>{
        if (skipIntro == false) {
            container.classList.add('animation3');
            description.classList.add('animation1');
        }
    }, 5000)

    setTimeout(()=>{
        if (skipIntro == false) {
            description.classList.add('animation2');
        }
    }, 5500)

    setTimeout(()=>{
        if (skipIntro == false) {
            descriptionTitle.classList.add('active');
            descriptionText.classList.add('active');
        }
    }, 6000)

    setTimeout(()=>{
        if (skipIntro == false) {
            descriptionTitle.classList.remove('active');
            descriptionText.classList.remove('active');
            descriptionTitle.classList.add('fade');
            descriptionText.classList.add('fade');
        }
    }, 26500)

    setTimeout(()=>{
        if (skipIntro == false) {
            description.classList.add('animation3');
        }
    }, 27500)

    setTimeout(()=>{
        if (skipIntro == false) {
            skipContainer.style.transition = '1s ease-in';
            page.style.top = '-100vh';
            skipContainer.style.top = '-100vh';
            main.style.top = '0px';
        }
    }, 28000)
})

function SkipWelcomePage() {
    skipIntro = true
    skipContainer.style.transition = '1s ease-in';
    page.style.top = '-100vh';
    skipContainer.style.top = '-100vh';
    main.style.top = '0px';
}